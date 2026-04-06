# Tahoe SLM Colab Training Pack

This folder provides a Colab-first training workflow for QLoRA fine-tuning.

## What you get
- `train_tahoe_qwen15b_qlora_colab.ipynb`: one-notebook training flow
- `evaluate_extractor.py`: validation metrics on `valid_pairs.json`

## Base + data
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Dataset: `hariharan5693/tahoe-synthetic-extraction-v1`

## Output model repo
Set in notebook variable `MODEL_REPO` (example):
- `hariharan5693/tahoe-qwen25-1p5b-extractor-lora-v1`

## Colab quick run
1. Open notebook in Colab.
2. Runtime -> Change runtime type -> GPU (T4/A100 if available).
3. Set HF token in notebook cell (`HF_TOKEN`).
4. Run all cells.
5. Notebook pushes LoRA adapter to your Hugging Face model repo.

## After training
Run `evaluate_extractor.py` in Colab/local to report:
- required-field pass rate
- field exact match rate
- JSON parse success rate

## Note
- This trains a LoRA adapter (recommended for cost/perf).
- Keep deterministic ETL as primary path in product.
