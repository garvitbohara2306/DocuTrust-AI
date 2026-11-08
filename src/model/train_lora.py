import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BASE = Path("data/training")
OUT = Path("models/docutrust-qwen-lora")
MAX_LEN = 512
EPOCHS = 3

assert torch.cuda.is_available(), "CUDA GPU not available"
torch.cuda.empty_cache()

def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

train_rows = load_jsonl(BASE / "doc002_sft_train.jsonl")
val_rows = load_jsonl(BASE / "doc002_sft_validation.jsonl")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def encode(row):
    user_text = row["instruction"] + "\\n\\n" + row["input"]

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=True,
        add_generation_prompt=True
    )["input_ids"]

    full = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": row["output"]}
        ],
        tokenize=True,
        add_generation_prompt=False
    )["input_ids"]

    common = 0
    while common < min(len(prompt), len(full)) and prompt[common] == full[common]:
        common += 1

    if common < len(prompt) - 3:
        raise ValueError(
            f"Chat template mismatch: prompt={len(prompt)}, common={common}"
        )

    answer_ids = full[common:]
    if not answer_ids:
        raise ValueError("No answer tokens found")

    answer_ids = answer_ids[:MAX_LEN]
    prompt_budget = MAX_LEN - len(answer_ids)
    prompt_ids = prompt[-prompt_budget:] if prompt_budget > 0 else []

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    return {"input_ids": input_ids, "labels": labels}

class SFTDataset(Dataset):
    def __init__(self, rows):
        self.items = [encode(r) for r in rows]
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        return self.items[i]

def collate(items):
    n = max(len(x["input_ids"]) for x in items)
    ids, masks, labels = [], [], []
    for x in items:
        pad = n - len(x["input_ids"])
        ids.append(x["input_ids"] + [tokenizer.pad_token_id] * pad)
        masks.append([1] * len(x["input_ids"]) + [0] * pad)
        labels.append(x["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }

train_ds = SFTDataset(train_rows)
val_ds = SFTDataset(val_rows)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate)

print("Loading base model on GPU...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16
).to("cuda")

model.config.use_cache = False
model = get_peft_model(model, LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
))
model.print_trainable_parameters()

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-4
)

def evaluate():
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss
            total += loss.item()
            count += 1
    model.train()
    return total / max(count, 1)

OUT.mkdir(parents=True, exist_ok=True)
for epoch in range(EPOCHS):
    model.train()
    total, count = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(train_loader, 1):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(**batch).loss
        (loss / 4).backward()
        if step % 4 == 0 or step == len(train_loader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total += loss.item()
        count += 1
    val_loss = evaluate()
    print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={total/count:.4f} | val_loss={val_loss:.4f}")

model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)
print("Adapter saved to:", OUT)
print("Peak GPU memory:", round(torch.cuda.max_memory_allocated()/1024**3, 2), "GB")
