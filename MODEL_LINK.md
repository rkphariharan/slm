# Model Link

- Trained hosted model (LoRA adapter):
  - https://huggingface.co/hariharan5693/tahoe-qwen25-3b-extractor-lora-v1

## Notes
- Base model family: Qwen2.5 (LoRA/QLoRA workflow)
- Training data source: synthetic dataset generated from OCI-shaped patterns
- For production app flow: deterministic ETL first, SLM fallback for messy uploads
