"""
layer_pruning_final.py
=====================================
実験まとめベスト構成:
  - NF4量子化
  - BIスコア下位6層: 完全削除（残差なし）
  - 偶数層のうちBI下位6層以外の13層: 残差補正あり（推論時のみVRAMロード）
  - KVキャッシュ有効

結果 (Qwen2.5-3B):
  ベースライン:  PPL 3.46 / 13~14 tok/s / 1.92GB
  本モデル:      PPL 3.63 / 19.1  tok/s / 待機1.22GB / 推論1.71GB

使い方:
  python layer_pruning_final.py --model /path/to/model
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


# ============================================================
# ベスト構成
# ============================================================

# BIスコア下位6層 → 完全削除（残差なし）
HARD_SKIP = [19, 20, 21, 22, 23, 24]

# 偶数層のうちHARD_SKIPと重複しない13層 → 残差補正あり
_EVEN = set(i for i in range(2, 34) if i % 2 == 0)
SOFT_SKIP = sorted(_EVEN - set(HARD_SKIP))
# = [2, 4, 6, 8, 10, 12, 14, 16, 18, 26, 28, 30, 32]


# ============================================================
# モデル
# ============================================================

class LayerPruningModel(nn.Module, GenerationMixin):
    """
    hard_skip_ids: 完全削除（残差なし）
    soft_skip_ids: 削除 + 残差補正あり（推論時のみVRAMロード）
    それ以外: 通常通り保持
    """

    _is_stateful = False
    _supports_cache_class = False
    main_input_name = "input_ids"

    def __init__(self, original_model,
                 hard_skip_ids: list[int],
                 soft_skip_ids: list[int],
                 cache_dir: str = "./pruning_cache",
                 alpha: float = 1.0):
        super().__init__()
        m = original_model.model

        self.config = original_model.config
        self.generation_config = GenerationConfig.from_model_config(
            original_model.config)
        self.device = next(original_model.parameters()).device

        self.embed_tokens = m.embed_tokens
        self.rotary_emb = m.rotary_emb
        self.hard_skip_ids = set(hard_skip_ids)
        self.soft_skip_ids = set(soft_skip_ids)
        self.all_skip_ids = self.hard_skip_ids | self.soft_skip_ids
        self.sorted_soft_skip_ids = sorted(soft_skip_ids)
        self.alpha = alpha
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.norm = m.norm
        self.lm_head = original_model.lm_head
        self.n_layers = len(m.layers)

        print(f"  完全削除 ({len(hard_skip_ids)}層): {sorted(hard_skip_ids)}")
        print(f"  残差補正 ({len(soft_skip_ids)}層): {sorted(soft_skip_ids)}")
        print(f"  保持     ({self.n_layers - len(self.all_skip_ids)}層): "
              f"{[i for i in range(self.n_layers) if i not in self.all_skip_ids]}")

        # inject_map: soft_skip層の残差をどの層に注入するか
        self.inject_map = {}
        for skip_id in soft_skip_ids:
            for candidate in range(skip_id + 1, self.n_layers):
                if candidate not in self.all_skip_ids:
                    self.inject_map[skip_id] = candidate
                    break

        self.inject_targets = {}
        for skip_id, inject_id in self.inject_map.items():
            self.inject_targets.setdefault(inject_id, []).append(skip_id)

        self.layer_id_map = {}
        new_idx = 0
        for i in range(self.n_layers):
            if i not in self.all_skip_ids:
                self.layer_id_map[i] = new_idx
                new_idx += 1

        # soft_skip層をディスクに保存
        print("  残差補正層をディスクに保存中...")
        for i, layer in enumerate(m.layers):
            if i in self.soft_skip_ids:
                path = self.cache_dir / f"layer_{i}.pt"
                if not path.exists():
                    torch.save(layer, path)
                    print(f"    層{i} 保存完了")
                else:
                    print(f"    層{i} キャッシュ済み")

        # 全削除層をVRAMから解放
        for i in range(self.n_layers):
            if i in self.all_skip_ids:
                m.layers[i] = None
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  削除後VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

        # 保持層のみVRAMに残す
        self.layers = nn.ModuleList([
            m.layers[i] for i in range(self.n_layers)
            if i not in self.all_skip_ids
        ])
        for new_idx, layer in enumerate(self.layers):
            if hasattr(layer, 'self_attn'):
                layer.self_attn.layer_idx = new_idx

        self.soft_layers = None
        self.skip_past_kv = None
        print(f"  構築完了 VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    def load_soft_layers(self):
        """推論開始時に残差補正層をVRAMにロード"""
        self.soft_layers = {}
        for new_idx, skip_id in enumerate(self.sorted_soft_skip_ids):
            path = self.cache_dir / f"layer_{skip_id}.pt"
            layer = torch.load(
                path, map_location=str(self.device), weights_only=False)
            if hasattr(layer, 'self_attn'):
                layer.self_attn.layer_idx = new_idx
            self.soft_layers[skip_id] = layer

    def release_soft_layers(self):
        """推論終了時に残差補正層をVRAMから解放"""
        if self.soft_layers is not None:
            del self.soft_layers
            self.soft_layers = None
            self.skip_past_kv = None
            gc.collect()
            torch.cuda.empty_cache()

    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, e): self.lm_head = e
    def get_input_embeddings(self): return self.embed_tokens
    def set_input_embeddings(self, e): self.embed_tokens = e
    def can_generate(self): return True

    def forward(self, input_ids, past_key_values=None,
                attention_mask=None, use_cache=True, **kwargs):
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
            # 完全削除: スキップのみ
            if orig_id in self.hard_skip_ids:
                continue

            # 残差補正あり削除: 入力を記録してスキップ
            if orig_id in self.soft_skip_ids:
                x_before_skip[orig_id] = x.clone()
                continue

            # 残差補正の注入
            if orig_id in self.inject_targets:
                for skip_id in self.inject_targets[orig_id]:
                    if skip_id in x_before_skip:
                        skip_layer = self.soft_layers[skip_id]
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
                        x = x + self.alpha * (skip_x - x_before_skip[skip_id])

            # 通常層
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
# ユーティリティ
# ============================================================

def generate(model, tokenizer, prompt, device, max_new=20):
    model.load_soft_layers()
    model.skip_past_kv = None
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated = inputs.input_ids.clone()
    past_kv = None
    with torch.no_grad():
        for step in range(max_new):
            out = model(
                generated if step == 0 else generated[:, -1:],
                past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    model.release_soft_layers()
    n = inputs.input_ids.shape[1]
    return tokenizer.decode(generated[0][n:], skip_special_tokens=True)


def compute_perplexity(model, tokenizer, texts, device):
    model.load_soft_layers()
    model.skip_past_kv = None
    total_loss, total_tokens = 0.0, 0
    for text in texts:
        model.skip_past_kv = None
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue
        with torch.no_grad():
            out = model(input_ids, past_key_values=None, use_cache=False)
        shift_logits = out.logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += shift_labels.numel()
    model.release_soft_layers()
    return float(torch.exp(torch.tensor(total_loss / total_tokens)))


def measure_tps(model, tokenizer, prompt, device, n_tokens=20, n_runs=3):
    model.load_soft_layers()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    start = time.perf_counter()
    for _ in range(n_runs):
        model.skip_past_kv = None
        generated = inputs.input_ids.clone()
        past_kv = None
        with torch.no_grad():
            for step in range(n_tokens):
                out = model(
                    generated if step == 0 else generated[:, -1:],
                    past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
    tps = n_tokens / ((time.perf_counter() - start) / n_runs)
    inference_vram = torch.cuda.memory_allocated() / 1024**3
    model.release_soft_layers()
    return tps, inference_vram


# ============================================================
# メイン
# ============================================================

def run(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"モデル: {model_name}")
    print("NF4量子化でロード中...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    original = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        quantization_config=bnb_config, device_map="cuda")
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
            original.generate(inputs.input_ids, max_new_tokens=20,
                              do_sample=False, pad_token_id=tokenizer.eos_token_id)
    base_tps = 20 / ((time.perf_counter() - start) / 3)
    print(f"  速度: {base_tps:.1f} tok/s  VRAM: {base_vram:.2f}GB")

    # モデル構築
    print(f"\nモデル構築中...")
    model = LayerPruningModel(
        original,
        hard_skip_ids=HARD_SKIP,
        soft_skip_ids=SOFT_SKIP,
        cache_dir="./pruning_cache",
        alpha=1.0,
    )
    model.eval()
    standby_vram = torch.cuda.memory_allocated() / 1024**3

    # 計測
    print("\nPPL計測中...")
    ppl = compute_perplexity(model, tokenizer, eval_texts, device)
    print("速度計測中...")
    tps, inference_vram = measure_tps(model, tokenizer, test_prompts[0], device)

    print(f"\n{'='*60}")
    print(f"結果")
    print(f"{'='*60}")
    print(f"  ベースライン:  PPL {3.46:.2f}  {base_tps:.1f} tok/s  {base_vram:.2f}GB")
    print(f"  本モデル:      PPL {ppl:.2f}  {tps:.1f} tok/s  待機{standby_vram:.2f}GB 推論{inference_vram:.2f}GB")
    print(f"  速度改善:      +{tps/base_tps*100-100:.0f}%")
    print(f"  待機VRAM削減:  -{base_vram-standby_vram:.2f}GB ({(base_vram-standby_vram)/base_vram*100:.0f}%)")

    print(f"\n{'='*60}")
    print(f"出力")
    print(f"{'='*60}")
    for prompt in test_prompts:
        result = generate(model, tokenizer, prompt, device)
        print(f"  [{prompt[:30]}] → {result[:45]}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "--model":
        print("使い方: python layer_pruning_final.py --model /path/to/model")
        sys.exit(1)
    run(sys.argv[2])
