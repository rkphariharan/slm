# Qwen Retraining V2 (Mixed: old synthetic + demo dataset mappings)

## Purpose
Continue fine-tuning your existing Qwen extractor adapter with mixed data:
- Old synthetic extraction samples (`4thaxis/synthetic`)
- New header-to-schema mapping samples generated from `4thaxis/demo/dataset`

This is for SLM-based upload mapping to DB table/column targets.

## New folders
- `slm/mixed_dataset_v2` -> dataset builder + mixed JSONL output
- `slm/training_v2` -> retraining script for GCP/Colab

## Step 1: Build mixed dataset (local)
From workspace root:

```powershell
& ".py-global/Scripts/python.exe" "slm/mixed_dataset_v2/build_mixed_dataset.py"
```

Expected outputs in `slm/mixed_dataset_v2/`:
- `train_chat_mixed_v2.jsonl`
- `valid_chat_mixed_v2.jsonl`
- `mixed_dataset_metadata.json`

## Step 2: Retrain on GCP/Colab
Upload the generated JSONL files and script to your GCP/Colab runtime, then run:

```bash
export HF_TOKEN="hf_xxx"
export TRAIN_FILE="/content/train_chat_mixed_v2.jsonl"
export VALID_FILE="/content/valid_chat_mixed_v2.jsonl"
export CONTINUE_FROM_MODEL="hariharan5693/tahoe-qwen25-3b-extractor-lora-v1"
export MODEL_REPO="hariharan5693/tahoe-qwen25-3b-extractor-lora-v2-mixed"
python gcp_retrain_qwen_mixed_v2.py
```

## Optional tuning knobs
- `EPOCHS` (default `1.5`)
- `LR` (default `5e-5`)
- `MAX_LEN` (default `2048`)
- `BATCH` (default `1`)
- `GRAD_ACC` (default `8`)

## Notes
- This script continues from your previous adapter (`CONTINUE_FROM_MODEL`) rather than restarting.
- Keep a new repo name for the retrained adapter (`MODEL_REPO` v2).
- Do not overwrite v1 until you validate extraction quality.
