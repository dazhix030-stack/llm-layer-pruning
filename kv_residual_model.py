"""
残差補正 + KVキャッシュ + 推論時のみVRAMロード
=====================================
待機時は削除層をディスクに保存
推論時だけVRAMにロードして終わったら解放

使い方:
  python kv_residual_model.py --model /home/hosokawa_daichi/models/Qwen2.5-3B
"""

import torch
import torch.nn as nn
import sys
import gc
import time
from pathlib import Path
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GenerationMixin,
                          GenerationConfig)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import DynamicCache


class KVResidualModel(nn.Module, GenerationMixin):

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
        self.sorted_skip_ids = sorted(skip_ids)
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

        # layer_id_mapを先に作る
        self.layer_id_map = {}
        new_idx = 0
        for i in range(self.n_layers):
            if i not in self.skip_ids:
                self.layer_id_map[i] = new_idx
                new_idx += 1

        # 削除層をディスクに保存
        print("  削除層をディスクに保存中...")
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

        # 通常層のlayer_idxを詰め直す
        for new_idx, layer in enumerate(self.layers):
            if hasattr(layer, 'self_attn'):
                layer.self_attn.layer_idx = new_idx

        # 削除層はディスクのみ (VRAMには乗せない)
        self.skip_layers = None  # 推論時にロード
        self.skip_past_kv = None
        print(f"  構築完了 VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    def load_skip_layers(self):
        """推論開始時に削除層をVRAMにロード"""
        self.skip_layers = {}
        for new_idx, skip_id in enumerate(self.sorted_skip_ids):
            path = self.cache_dir / f"layer_{skip_id}.pt"
            layer = torch.load(
                path, map_location=str(self.device), weights_only=False)
            if hasattr(layer, 'self_attn'):
                layer.self_attn.layer_idx = new_idx
            self.skip_layers[skip_id] = layer

    def release_skip_layers(self):
        """推論終了時に削除層をVRAMから解放"""
        if self.skip_layers is not None:
            del self.skip_layers
            self.skip_layers = None
            self.skip_past_kv = None
            gc.collect()
            torch.cuda.empty_cache()

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

    def forward(self, input_ids,
                past_key_values=None,
                attention_mask=None,
                use_cache=True,
                **kwargs):
        device = input_ids.device

        if past_key_values is None:
            past_key_values = DynamicCache()
            self.skip_past_kv = DynamicCache()

        past_len = past_key_values.get_seq_length()
        seq_len = input_ids.shape[1]

        x = self.embed_tokens(input_ids)
        position_ids = torch.arange(
            past_len, past_len + seq_len, device=device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)

        x_before_skip = {}

        for orig_id in range(self.n_layers):
            if orig_id in self.skip_ids:
                x_before_skip[orig_id] = x.clone()
                continue

            if orig_id in self.inject_targets:
                for skip_id in self.inject_targets[orig_id]:
                    if skip_id in x_before_skip:
                        skip_layer = self.skip_layers[skip_id]
                        with torch.no_grad():
                            skip_out = skip_layer(
                                x_before_skip[skip_id],
                                attention_mask=None,
                                position_ids=position_ids,
                                past_key_values=self.skip_past_kv,
                                use_cache=True,
                                cache_position=None,
                                position_embeddings=position_embeddings,
                            )
                        skip_x = skip_out[0] if isinstance(skip_out, tuple) else skip_out
                        residual = skip_x - x_before_skip[skip_id]
                        x = x + self.alpha * residual

            new_idx = self.layer_id_map[orig_id]
            layer_out = self.layers[new_idx](
                x,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            x = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        x = self.norm(x)
        logits = self.lm_head(x)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
        )


# ============================================================
# 生成 (手動ループ)
# ============================================================

def generate_text(model, tokenizer, prompt, device, max_new=15):
    # 推論開始: 削除層をVRAMにロード
    model.load_skip_layers()
    model.skip_past_kv = None

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    generated = input_ids.clone()
    past_kv = None

    with torch.no_grad():
        for step in range(max_new):
            if step == 0:
                out = model(generated, past_key_values=None, use_cache=True)
            else:
                out = model(generated[:, -1:],
                           past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    # 推論終了: 削除層をVRAMから解放
    model.release_skip_layers()

    n = input_ids.shape[1]
    return tokenizer.decode(generated[0][n:], skip_special_tokens=True)


def compute_perplexity(model, tokenizer, texts, device):
    model.load_skip_layers()
    model.skip_past_kv = None

    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        model.skip_past_kv = None
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue
        with torch.no_grad():
            out = model(input_ids, past_key_values=None, use_cache=False)
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

    model.release_skip_layers()
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
    print(f"  速度: {base_tps:.1f} tok/s  VRAM: {base_vram:.2f}GB")

    print()
    print("=" * 60)
    print("ベースライン出力")
    print("=" * 60)
    for prompt in test_prompts:
        result = generate_text(original, tokenizer, prompt, device) \
            if False else None
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = original.generate(
                inputs.input_ids, max_new_tokens=20,
                do_sample=False, pad_token_id=tokenizer.eos_token_id)
        n = inputs.input_ids.shape[1]
        print(f"  [{prompt[:30]}] → "
              f"{tokenizer.decode(out[0][n:], skip_special_tokens=True)[:45]}")

    # KV残差補正モデル
    skip_ids = [i for i in range(2, 33) if i % 2 == 0]
    print(f"\nKV残差補正モデル構築中 ({len(skip_ids)}層削除)...")

    kv_model = KVResidualModel(
        original,
        skip_ids=skip_ids,
        cache_dir="./direct_cache",
        alpha=1.0,
    )
    kv_model.eval()

    del original
    gc.collect()
    torch.cuda.empty_cache()

    standby_vram = torch.cuda.memory_allocated() / 1024**3
    print(f"待機時VRAM: {standby_vram:.2f}GB")

    # PPL
    print("\nPPL計測中...")
    ppl = compute_perplexity(kv_model, tokenizer, eval_texts, device)
    print(f"PPL: {ppl:.2f}")

    # 速度計測
    print("\n速度計測中...")
    kv_model.load_skip_layers()
    start = time.perf_counter()
    for _ in range(3):
        kv_model.skip_past_kv = None
        inputs = tokenizer(test_prompts[0], return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        generated = input_ids.clone()
        past_kv = None
        with torch.no_grad():
            for step in range(20):
                if step == 0:
                    out = kv_model(generated, past_key_values=None, use_cache=True)
                else:
                    out = kv_model(generated[:, -1:],
                                  past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
    kv_tps = 20 / ((time.perf_counter() - start) / 3)
    inference_vram = torch.cuda.memory_allocated() / 1024**3
    kv_model.release_skip_layers()

    print()
    print("=" * 60)
    print("結果")
    print("=" * 60)
    print(f"  ベースライン:      {base_tps:.1f} tok/s  {base_vram:.2f}GB")
    print(f"  KV残差補正 待機時: -         {standby_vram:.2f}GB")
    print(f"  KV残差補正 推論時: {kv_tps:.1f} tok/s  {inference_vram:.2f}GB")
    print(f"  PPL: {ppl:.2f}")
    print(f"  待機時VRAM削減: {base_vram - standby_vram:.2f}GB "
          f"({(base_vram-standby_vram)/base_vram*100:.1f}%)")

    print()
    print("=" * 60)
    print("出力")
    print("=" * 60)
    for prompt in test_prompts:
        result = generate_text(kv_model, tokenizer, prompt, device)
        print(f"  [{prompt[:30]}] → {result[:45]}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "--model":
        print("使い方: python kv_residual_model.py "
              "--model /home/hosokawa_daichi/models/Qwen2.5-3B")
        sys.exit(1)

    run(sys.argv[2])