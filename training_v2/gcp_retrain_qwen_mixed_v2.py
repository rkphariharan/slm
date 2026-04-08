import inspect
import os

import torch
from datasets import load_dataset
from huggingface_hub import login
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU not detected. In Colab go to Runtime > Change runtime type > Hardware accelerator = GPU, "
        "then restart runtime and rerun all cells."
    )

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN or not HF_TOKEN.startswith("hf_"):
    raise ValueError("Set HF_TOKEN as environment variable.")

login(token=HF_TOKEN)
print("HF login ok")

TRAIN_FILE = os.environ.get("TRAIN_FILE", "slm/mixed_dataset_v2/train_chat_mixed_v2.jsonl")
VALID_FILE = os.environ.get("VALID_FILE", "slm/mixed_dataset_v2/valid_chat_mixed_v2.jsonl")

CONTINUE_FROM_MODEL = os.environ.get(
    "CONTINUE_FROM_MODEL",
    "hariharan5693/tahoe-qwen25-3b-extractor-lora-v1",
)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "tahoe_qwen_mixed_v2_out")
MODEL_REPO = os.environ.get("MODEL_REPO", "hariharan5693/tahoe-qwen25-3b-extractor-lora-v2-mixed")

MAX_LEN = int(os.environ.get("MAX_LEN", "2048"))
EPOCHS = float(os.environ.get("EPOCHS", "1.5"))
LR = float(os.environ.get("LR", "5e-5"))
BATCH = int(os.environ.get("BATCH", "1"))
GRAD_ACC = int(os.environ.get("GRAD_ACC", "8"))

train_ds = load_dataset("json", data_files=TRAIN_FILE, split="train")
valid_ds = load_dataset("json", data_files=VALID_FILE, split="train")
print("train", len(train_ds), "valid", len(valid_ds))

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CONTINUE_FROM_MODEL,
    max_seq_length=MAX_LEN,
    dtype=None,
    load_in_4bit=True,
)

if not hasattr(model, "peft_config"):
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

print("model ready")


def to_text(example):
    msgs = example["messages"]
    txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return {"text": txt}


train_text = train_ds.map(to_text, remove_columns=train_ds.column_names)
valid_text = valid_ds.map(to_text, remove_columns=valid_ds.column_names)

ta_kwargs = dict(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=20,
    eval_steps=100,
    save_steps=100,
    fp16=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    report_to="none",
    do_eval=True,
)

if "evaluation_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
    ta_kwargs["evaluation_strategy"] = "steps"
else:
    ta_kwargs["eval_strategy"] = "steps"

train_args = TrainingArguments(**ta_kwargs)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_text,
    eval_dataset=valid_text,
    dataset_text_field="text",
    max_seq_length=MAX_LEN,
    args=train_args,
)

trainer.train()

model.push_to_hub(MODEL_REPO, token=HF_TOKEN)
tokenizer.push_to_hub(MODEL_REPO, token=HF_TOKEN)

print("retrained model pushed:", f"https://huggingface.co/{MODEL_REPO}")
