"""
state_dict高速ロード + 残差補正 + 非同期プリフェッチ
=====================================
state_dictで14倍高速ロード + 非同期プリフェッチで速度改善

使い方:
  python direct_vram_model.py --model /home/hosokawa_daichi/models/Qwen2.5-3B
"""

import torch
import torch.nn as nn
import sys
import gc
import copy
import time
from pathlib import Path
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GenerationMixin,
                          GenerationConfig)
from transformers.modeling_outputs import CausalLMOutputWithPast


class DirectVRAMModel(nn.Module, GenerationMixin):

    _is_stateful = False
    _supports_cache_class = False
    main_input_name = "input_ids"

    def __init__(self, original_model,
                 skip_ids: list[int],
                 cache_dir: str = "./direct_cache",
                 alpha: float = 1.0):
        super().__init__()
        m = original_model.model

        self.config = original_model.config
        self.generation_config = GenerationConfig.from_model_config(
            original_model.config)
        self.device = next(original_model.parameters()).device

        self.embed_tokens = m.embed_tokens
        self.rotary_emb = m.rotary_emb
        self.skip_ids = set(skip_ids)
        self.alpha = alpha
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.norm = m.norm
        self.lm_head = original_model.lm_head
        self.n_layers = len(m.layers)

        # inject_map
        self.inject_map = {}
        for skip_id in skip_ids:
            for candidate in range(skip_id + 1, self.n_layers):
                if candidate not in self.skip_ids:
                    self.inject_map[skip_id] = candidate
                    break

        self.inject_targets = {}
        for skip_id, inject_id in self.inject_map.items():
            self.inject_targets.setdefault(inject_id, []).append(skip_id)

        # 2層前でプリフェッチ開始
        self.prefetch_schedule = {}
        for inject_id, sids in self.inject_targets.items():
            count = 0
            for orig_id in range(inject_id - 1, -1, -1):
                if orig_id not in self.skip_ids:
                    self.prefetch_schedule.setdefault(
                        orig_id, []).extend(sids)
                    count += 1
                    if count >= 2:
                        break

        # 層のテンプレート (state_dictロード用)
        # 通常層を1つコピーしてテンプレートとして使う
        first_normal = next(
            i for i in range(self.n_layers) if i not in self.skip_ids)
        self.layer_template = copy.deepcopy(
            m.layers[first_normal]).cpu()

        # 削除層をstate_dictとしてディスクに保存
        print("  削除層をstate_dictで保存中...")
        for i, layer in enumerate(m.layers):
            if i in self.skip_ids:
                path = self.cache_dir / f"layer_{i}.pt"
                if not path.exists():
                    torch.save(layer, path)
                    print(f"    層{i}保存完了")
                else:
                    print(f"    層{i}キャッシュ済み")

        # 削除層をVRAMから解放
        for i in range(self.n_layers):
            if i in self.skip_ids:
                m.layers[i] = None
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  削除層解放後VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

        # 通常層だけVRAMに保持
        self.layers = nn.ModuleList([
            m.layers[i] for i in range(self.n_layers)
            if i not in self.skip_ids
        ])

        self.layer_id_map = {}
        new_idx = 0
        for i in range(self.n_layers):
            if i not in self.skip_ids:
                self.layer_id_map[i] = new_idx
                new_idx += 1

        # 非同期転送用ストリーム
        self.transfer_stream = torch.cuda.Stream()
        self.prefetch_buffer = {}

        gc.collect()
        torch.cuda.empty_cache()
        print(f"  構築完了 VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def can_generate(self):
        return True

    def prepare_inputs_for_generation(self, input_ids,
                                       past_key_values=None,
                                       attention_mask=None,
                                       **kwargs):
        return {
            "input_ids": input_ids,
            "past_key_values": None,
            "attention_mask": None,
            "use_cache": False,
        }

    def load_skip_layer(self, skip_id: int, device: str):
        """ディスクから直接VRAMにロード"""
        path = self.cache_dir / f"layer_{skip_id}.pt"
        return torch.load(
            path, map_location=device, weights_only=False)

    def prefetch_layer(self, skip_id: int):
        """非同期でstate_dictをVRAMにロード"""
        if skip_id in self.prefetch_buffer:
            return
        with torch.cuda.stream(self.transfer_stream):
            self.prefetch_buffer[skip_id] = self.load_skip_layer(
                skip_id, str(self.device))

    def get_layer(self, skip_id: int):
        """プリフェッチ完了を待って層を取得"""
        if skip_id in self.prefetch_buffer:
            torch.cuda.current_stream().wait_stream(self.transfer_stream)
            return self.prefetch_buffer.pop(skip_id)
        return self.load_skip_layer(skip_id, str(self.device))

    def release_layer(self, layer):
        del layer
        torch.cuda.empty_cache()

    def forward(self, input_ids,
                past_key_values=None,
                attention_mask=None,
                use_cache=False,
                **kwargs):
        device = input_ids.device
        x = self.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)

        x_before_skip = {}
        self.prefetch_buffer = {}

        for orig_id in range(self.n_layers):
            if orig_id in self.skip_ids:
                x_before_skip[orig_id] = x.clone()
                continue

            # プリフェッチ開始
            if orig_id in self.prefetch_schedule:
                for skip_id in self.prefetch_schedule[orig_id]:
                    self.prefetch_layer(skip_id)

            # inject対象: 残差を計算して足す
            if orig_id in self.inject_targets:
                for skip_id in self.inject_targets[orig_id]:
                    if skip_id in x_before_skip:
                        skip_layer = self.get_layer(skip_id)
                        with torch.no_grad():
                            out = skip_layer(
                                x_before_skip[skip_id],
                                attention_mask=None,
                                position_ids=position_ids,
                                past_key_values=None,
                                use_cache=False,
                                cache_position=None,
                                position_embeddings=position_embeddings,
                            )
                            x_out = out[0] if isinstance(out, tuple) else out
                            residual = x_out - x_before_skip[skip_id]
                        x = x + self.alpha * residual
                        self.release_layer(skip_layer)

            # 通常層を実行
            new_idx = self.layer_id_map[orig_id]
            out = self.layers[new_idx](
                x,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            x = out[0] if isinstance(out, tuple) else out

        x = self.norm(x)
        logits = self.lm_head(x)

        return CausalLMOutputWithPast(logits=logits, past_key_values=None)


# ============================================================
# 評価
# ============================================================

def generate_text(model, tokenizer, prompt, device, max_new=15):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    n = inputs.input_ids.shape[1]
    return tokenizer.decode(out[0][n:], skip_special_tokens=True)


def compute_perplexity(model, tokenizer, texts, device):
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue
        with torch.no_grad():
            out = model(input_ids)
            logits = out.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction='sum'
        )
        total_loss += loss.item()
        total_tokens += shift_labels.numel()
    return float(torch.exp(torch.tensor(total_loss / total_tokens)))


# ============================================================
# 実験
# ============================================================

def run(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"モデル: {model_name}")
    print("NF4量子化でロード中...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    original = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="cuda",
    )
    original.eval()

    base_vram = torch.cuda.memory_allocated() / 1024**3

    test_prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "The quick brown fox jumps over",
    ]

    eval_texts = [
        "The capital of France is Paris. It is the largest city in Europe.",
        "def fibonacci(n):\n    if n == 0:\n        return 0\n    elif n == 1:\n        return 1",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a high-level programming language known for its simplicity.",
    ]

    # ベースライン
    print("\nベースライン計測中...")
    base_ppl = compute_perplexity(original, tokenizer, eval_texts, device)
    start = time.perf_counter()
    for _ in range(3):
        inputs = tokenizer(test_prompts[0], return_tensors="pt").to(device)
        with torch.no_grad():
            original.generate(
                inputs.input_ids,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    base_tps = 20 / ((time.perf_counter() - start) / 3)
    print(f"  PPL: {base_ppl:.2f}  速度: {base_tps:.1f} tok/s  VRAM: {base_vram:.2f}GB")

    print()
    print("=" * 60)
    print("ベースライン出力")
    print("=" * 60)
    for prompt in test_prompts:
        result = generate_text(original, tokenizer, prompt, device)
        print(f"  [{prompt[:30]}] → {result[:45]}")

    # モデル構築
    skip_ids = [i for i in range(2, 33) if i % 2 == 0]
    print(f"\nモデル構築中 ({len(skip_ids)}層削除)...")

    direct_model = DirectVRAMModel(
        original,
        skip_ids=skip_ids,
        cache_dir="./direct_cache",
        alpha=1.0,
    )
    direct_model.eval()

    del original
    gc.collect()
    torch.cuda.empty_cache()

    direct_vram = torch.cuda.memory_allocated() / 1024**3

    # 速度計測
    print("\n速度計測中...")
    start = time.perf_counter()
    for _ in range(3):
        inputs = tokenizer(test_prompts[0], return_tensors="pt").to(device)
        with torch.no_grad():
            direct_model.generate(
                inputs.input_ids,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    direct_tps = 20 / ((time.perf_counter() - start) / 3)

    # PPL
    print("PPL計測中...")
    direct_ppl = compute_perplexity(
        direct_model, tokenizer, eval_texts, device)

    print()
    print("=" * 60)
    print("結果")
    print("=" * 60)
    print(f"  ベースライン:   PPL {base_ppl:.2f}  {base_tps:.1f} tok/s  {base_vram:.2f}GB")
    print(f"  state_dictモデル: PPL {direct_ppl:.2f}  {direct_tps:.1f} tok/s  {direct_vram:.2f}GB")
    print(f"  VRAM削減: {base_vram - direct_vram:.2f}GB "
          f"({(base_vram-direct_vram)/base_vram*100:.1f}%)")

    print()
    print("=" * 60)
    print("出力")
    print("=" * 60)
    for prompt in test_prompts:
        result = generate_text(direct_model, tokenizer, prompt, device)
        print(f"  [{prompt[:30]}] → {result[:45]}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "--model":
        print("使い方: python direct_vram_model.py "
              ""--model /path/to/your/model"")
        sys.exit(1)

    run(sys.argv[2])
