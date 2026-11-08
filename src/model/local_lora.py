
from pathlib import Path
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = ROOT / "models" / "docutrust-qwen-lora"


# ============================================================
# LOAD FINE-TUNED MODEL
# ============================================================

def load_local_model():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this model.")

    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found: {ADAPTER_PATH}"
        )

    print("Loading Qwen tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading Qwen base model on GPU...")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to("cuda")

    print("Loading trained LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_PATH)
    )

    model.eval()

    print("Fine-tuned Qwen model ready!")

    return tokenizer, model


# ============================================================
# CITATION VALIDATION
# ============================================================

def validate_citations(answer, allowed_pages):
    """
    Validate citation presence and page membership.

    This does NOT prove that a citation semantically
    supports the claim.
    """

    citations = re.findall(
        r"\[Page\s+(\d+)\]",
        answer,
        re.IGNORECASE
    )

    if not citations:
        return False, "No page citations found."

    cited_pages = {int(page) for page in citations}

    invalid_pages = cited_pages - set(allowed_pages)

    if invalid_pages:
        return False, (
            f"Invalid page citations: {sorted(invalid_pages)}"
        )

    return True, "Citation page membership passed."


# ============================================================
# MODEL GENERATION HELPER
# ============================================================

def run_generation(
    messages,
    tokenizer,
    model,
    max_new_tokens=180
):

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

    with torch.inference_mode():

        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(
        output[0][input_ids.shape[-1]:],
        skip_special_tokens=True
    ).strip()

    if not answer:
        raise RuntimeError("Qwen generated an empty answer.")

    return answer


# ============================================================
# GENERATE DOCUMENT-GROUNDED ANSWER
# ============================================================

def generate_answer(question, evidence, tokenizer, model):

    if not evidence:
        return (
            "INSUFFICIENT_EVIDENCE: "
            "No relevant document passages were retrieved."
        )

    # --------------------------------------------------------
    # Prepare document evidence
    # --------------------------------------------------------

    context_parts = []

    for item in evidence:

        page_number = int(item["page_number"])
        chunk_id = item.get("chunk_id", "Unknown")

        context_parts.append(
            f"[Page {page_number} | {chunk_id}]\n"
            f"{item['text']}"
        )

    context = "\n\n".join(context_parts)

    allowed_pages = sorted({
        int(item["page_number"])
        for item in evidence
    })

    allowed_pages_text = ", ".join(
        str(page) for page in allowed_pages
    )

    # --------------------------------------------------------
    # System prompt
    # --------------------------------------------------------

    system_prompt = (
        "You are DocuTrust AI, a strict document-grounded assistant.\n\n"

        "RULES:\n"
        "1. Use ONLY the supplied document evidence.\n"
        "2. Answer the exact question directly.\n"
        "3. Never use outside knowledge.\n"
        "4. Never invent facts or citations.\n"
        "5. Every factual statement must end with a page citation.\n"
        "6. Citation format must be exactly [Page 14].\n"
        "7. Use only the allowed page numbers.\n"
        "8. If evidence is insufficient, respond with "
        "INSUFFICIENT_EVIDENCE: followed by a short explanation.\n"
        "9. Keep the answer concise.\n"
        "10. Do not write a factual answer without citations.\n\n"

        "EXAMPLE:\n"
        "WCAG 2.0 is listed as an accessibility standard. [Page 14]\n"
        "PDF/UA is formally associated with ISO 14289. [Page 14]\n\n"

        f"ALLOWED PAGE NUMBERS: {allowed_pages_text}"
    )

    user_prompt = (
        f"DOCUMENT EVIDENCE:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Answer using only the supplied evidence. "
        "Cite every factual statement."
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    # --------------------------------------------------------
    # First generation attempt
    # --------------------------------------------------------

    print("Generating document-grounded answer...")

    answer = run_generation(
        messages,
        tokenizer,
        model,
        max_new_tokens=180
    )

    valid, reason = validate_citations(
        answer,
        allowed_pages
    )

    if valid:
        print("Citation page membership passed.")
        return answer

    # --------------------------------------------------------
    # One citation repair attempt
    # --------------------------------------------------------

    print(f"Citation validation failed: {reason}")
    print("Attempting one citation repair...")

    repair_messages = [
        {
            "role": "system",
            "content": (
                "You are a document-grounded answer editor.\n"
                "You must preserve the original factual meaning.\n"
                "Use only the supplied evidence.\n"
                "Add a correct [Page N] citation to every factual "
                "statement, using only allowed page numbers.\n"
                "Do not invent citations.\n"
                "If the evidence cannot support the answer, respond "
                "with INSUFFICIENT_EVIDENCE: and explain briefly.\n"
                "Return only the corrected answer."
            )
        },
        {
            "role": "user",
            "content": (
                f"QUESTION:\n{question}\n\n"
                f"ALLOWED PAGE NUMBERS:\n{allowed_pages_text}\n\n"
                f"DOCUMENT EVIDENCE:\n{context}\n\n"
                f"ORIGINAL ANSWER:\n{answer}\n\n"
                "Correct the answer and add supporting page citations."
            )
        }
    ]

    repaired_answer = run_generation(
        repair_messages,
        tokenizer,
        model,
        max_new_tokens=180
    )

    repaired_valid, repair_reason = validate_citations(
        repaired_answer,
        allowed_pages
    )

    if repaired_valid:
        print("Citation repair passed page membership validation.")
        return repaired_answer

    # --------------------------------------------------------
    # Fail safely
    # --------------------------------------------------------

    print(f"Citation repair failed: {repair_reason}")

    return (
        "UNVERIFIED — Citation validation failed after one retry.\n\n"
        + repaired_answer
    )
