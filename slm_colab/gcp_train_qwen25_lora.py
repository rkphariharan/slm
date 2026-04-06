import inspect
import os

from datasets import load_dataset
from huggingface_hub import login
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN or not HF_TOKEN.startswith("hf_"):
    raise ValueError("Set HF_TOKEN as an environment variable before running.")

login(token=HF_TOKEN)
print("HF login ok")

TRAIN_URL = "https://huggingface.co/datasets/hariharan5693/tahoe-synthetic-extraction-v1/resolve/main/train_chat.jsonl"
VALID_URL = "https://huggingface.co/datasets/hariharan5693/tahoe-synthetic-extraction-v1/resolve/main/valid_chat.jsonl"

train_ds = load_dataset("json", data_files=TRAIN_URL, split="train")
valid_ds = load_dataset("json", data_files=VALID_URL, split="train")

print("train", len(train_ds), "valid", len(valid_ds))

BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
MAX_LEN = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_LEN,
    dtype=None,
    load_in_4bit=True,
)

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
    output_dir="tahoe_qwen15b_lora_out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-4,
    logging_steps=20,
    eval_steps=100,
    save_steps=100,
    fp16=True,
    optim="adamw_torch",
    lr_scheduler_type="linear",
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

MODEL_REPO = "hariharan5693/tahoe-qwen25-3b-extractor-lora-v1"
model.push_to_hub(MODEL_REPO, token=HF_TOKEN)
tokenizer.push_to_hub(MODEL_REPO, token=HF_TOKEN)
print("https://huggingface.co/" + MODEL_REPO)
