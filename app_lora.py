
from pathlib import Path
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[0]

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = ROOT / "models" / "docutrust-qwen-lora"


# ============================================================
# LOAD MODEL
# ============================================================

def load_local_model():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found: {ADAPTER_PATH}"
        )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to("cuda")

    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_PATH)
    )

    model.eval()

    print("DocuTrust Qwen LoRA model loaded.")

    return tokenizer, model


# ============================================================
# GENERATION
# ============================================================

def run_generation(messages, tokenizer, model, max_tokens=180):

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
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(
        output[0][input_ids.shape[-1]:],
        skip_special_tokens=True
    ).strip()

    if not answer:
        raise RuntimeError("Model generated an empty answer.")

    return answer


# ============================================================
# CITATION VALIDATION
# ============================================================

def validate_citations(answer, allowed_pages):

    citations = re.findall(
        r"\[Page\s+(\d+)\]",
        answer,
        re.IGNORECASE
    )

    if not citations:
        return False

    cited_pages = {int(page) for page in citations}

    return cited_pages.issubset(set(allowed_pages))


# ============================================================
# CONSERVATIVE LEXICAL CITATION FALLBACK
# ============================================================

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "and", "or", "of", "in", "on", "to", "for",
    "with", "from", "that", "this", "these", "those",
    "what", "which", "who", "how", "document",
    "listed", "section", "formally", "called",
    "following", "according", "mentioned"
}


def content_terms(text):

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )

    return {
        word for word in words
        if word not in STOPWORDS and len(word) > 1
    }


def find_lexical_support(answer, evidence):
    """
    Attach a page citation only when the answer's
    meaningful terms have strong lexical coverage
    in a single retrieved evidence chunk.

    This is not semantic verification.
    """

    answer_terms = content_terms(answer)

    if len(answer_terms) < 2:
        return None

    best_page = None
    best_score = 0.0

    for item in evidence:

        source_terms = content_terms(item["text"])

        matched_terms = answer_terms.intersection(
            source_terms
        )

        score = len(matched_terms) / len(answer_terms)

        if score > best_score:
            best_score = score
            best_page = int(item["page_number"])

    # Conservative threshold.
    if best_score >= 0.80:
        return best_page

    return None


# ============================================================
# DOCUMENT-GROUNDED ANSWER
# ============================================================

def generate_answer(question, evidence, tokenizer, model):

    if not evidence:
        return (
            "INSUFFICIENT_EVIDENCE: "
            "No relevant document passages were retrieved."
        )

    context = "\n\n".join(
        f"[Page {item['page_number']} | {item['chunk_id']}]\n"
        f"{item['text']}"
        for item in evidence
    )

    allowed_pages = sorted({
        int(item["page_number"])
        for item in evidence
    })

    allowed_text = ", ".join(map(str, allowed_pages))

    system_prompt = (
        "You are DocuTrust AI, a document-grounded assistant.\n\n"
        "RULES:\n"
        "1. Answer using ONLY supplied document evidence.\n"
        "2. Answer the exact question.\n"
        "3. Never invent facts.\n"
        "4. Every factual statement must end with a citation.\n"
        "5. Citation format: [Page 14].\n"
        "6. Only cite pages from the supplied evidence.\n"
        "7. If evidence is insufficient, respond with "
        "INSUFFICIENT_EVIDENCE: followed by an explanation.\n"
        "8. Keep answers concise.\n\n"
        f"ALLOWED PAGES: {allowed_text}"
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": (
                f"DOCUMENT EVIDENCE:\n{context}\n\n"
                f"QUESTION:\n{question}\n\n"
                "Answer with supporting page citations."
            )
        }
    ]

    # First generation
    answer = run_generation(
        messages,
        tokenizer,
        model
    )

    if answer.upper().startswith("INSUFFICIENT_EVIDENCE:"):
        return answer

    if validate_citations(answer, allowed_pages):
        return answer

    print("Citation missing or invalid. Trying repair.")

    # One repair attempt
    repair_messages = [
        {
            "role": "system",
            "content": (
                "Correct the answer using only the supplied evidence. "
                "Add [Page N] citations to every factual statement. "
                "Use only allowed pages. Do not invent citations. "
                "Return only the corrected answer."
            )
        },
        {
            "role": "user",
            "content": (
                f"QUESTION:\n{question}\n\n"
                f"ALLOWED PAGES:\n{allowed_text}\n\n"
                f"EVIDENCE:\n{context}\n\n"
                f"ORIGINAL ANSWER:\n{answer}"
            )
        }
    ]

    repaired = run_generation(
        repair_messages,
        tokenizer,
        model
    )

    if repaired.upper().startswith("INSUFFICIENT_EVIDENCE:"):
        return repaired

    if validate_citations(repaired, allowed_pages):
        return repaired

    print("Repair failed. Checking lexical source support.")

    # Conservative deterministic fallback
    supported_page = find_lexical_support(
        repaired,
        evidence
    )

    if supported_page is not None:

        clean_answer = re.sub(
            r"\[Page\s+\d+\]",
            "",
            repaired,
            flags=re.IGNORECASE
        ).strip()

        clean_answer = clean_answer.rstrip(" .")

        print(
            f"Lexical support found in retrieved Page "
            f"{supported_page}."
        )

        return f"{clean_answer}. [Page {supported_page}]"

    # Fail safely
    return (
        "UNVERIFIED — No valid citation could be established.\n\n"
        + repaired
    )
