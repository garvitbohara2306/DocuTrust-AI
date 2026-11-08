import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[2]
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = ROOT / "models" / "docutrust-qwen-lora"

print("CUDA:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available")

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE)

print("Loading base Qwen model on GPU...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to("cuda")

print("Loading trained LoRA adapter...")
model = PeftModel.from_pretrained(base_model, str(ADAPTER))
model.eval()

messages = [
    {
        "role": "system",
        "content": (
            "Answer only from the supplied evidence. "
            "Do not invent facts. Cite factual claims as [Page N]."
        )
    },
    {
        "role": "user",
        "content": (
            "Evidence:\n"
            "[Page 14] An accessible PDF is a PDF document that "
            "can be accessed and used by persons with disabilities. "
            "The document lists WCAG 2.0 and PDF/UA standards.\n\n"
            "Question: What standards are listed in the document?"
        )
    }
]

encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

input_ids = encoded["input_ids"].to("cuda")
attention_mask = encoded.get("attention_mask")
if attention_mask is not None:
    attention_mask = attention_mask.to("cuda")

print("\nGenerating test answer...")
with torch.inference_mode():
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

answer = tokenizer.decode(
    output[0][input_ids.shape[-1]:],
    skip_special_tokens=True
)

print("\n========== LORA TEST RESULT ==========")
print(answer)
print("\nGPU memory allocated:", round(
    torch.cuda.memory_allocated() / 1024**3, 2
), "GB")
print("\nLORA GPU TEST COMPLETED")
