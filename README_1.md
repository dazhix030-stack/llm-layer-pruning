# LLM Layer Pruning with Residual Correction

NF4量子化 + 層削除 + 残差補正を組み合わせ、精度を維持したままVRAMを削減するLLM軽量化の実験。

## 概要

Transformer の中間層を最大44%削除しながら、削除した層の出力を残差として後続層に補正注入することで、PPLをほぼ維持したままメモリ使用量を大幅に削減する。

## 主な実験結果

| 設定 | VRAM | PPL | 速度 |
|------|------|-----|------|
| 3B ベースライン (float16) | 5.85GB | 3.04 | — |
| 3B 偶数層16層削除 + 残差補正 | 1.95GB | 3.04 | — |
| 3B NF4 ベースライン | 1.92GB | 3.46 | 14 tok/s |
| 3B NF4 + 層削除 + 残差補正 | 1.92GB | **3.43** | 6 tok/s |
| 3B NF4 + ディスクオフロード | **1.34GB** | 3.43 | 1.5 tok/s |

**VRAM 30%削減（ディスクオフロード時）、PPL変化 -0.03**

## 重要な発見

- **残差を層の「前」に足すことが鍵**: 「後」に足すと効果なし。1行の変更で劇的改善
- **NF4量子化後も残差補正は有効**
- **L0-1が緩衝地帯**: 超下位層への直接介入が崩壊の主因
- **ボトルネックはKVキャッシュ**: 計算量自体は変わらないため速度低下が課題

## アーキテクチャ

```
通常の推論:
  Input → Layer0 → Layer1 → ... → LayerN → Output

残差補正あり（Layer2を削除した場合）:
  Input → Layer0 → Layer1 → [Layer2をスキップ]
                                     ↓
                           Layer2を別途実行して残差を取得
                           residual = Layer2(x) - x
                                     ↓
                   Layer3入力に残差を加算して補正 → ... → Output
```

## 動作環境

- Python 3.12
- PyTorch 2.x
- transformers
- bitsandbytes（NF4量子化）
- GPU: RTX 4050 (6GB VRAM)

## 実行方法

```bash
python kv_residual_model.py --model /path/to/Qwen2.5-3B
```

## 未解決課題

- メモリ削減・速度・精度の三立（残差補正に削除層の実行が必要なため計算量は変わらない）
- KVキャッシュと残差補正の両立
- llama.cpp での実装（Python API から層単位の残差を取り出せない）
※ KVキャッシュ有効化により推論速度向上、VRAMは若干増加
