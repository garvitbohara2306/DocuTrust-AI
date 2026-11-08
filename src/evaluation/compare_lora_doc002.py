import csv
import gc
import json
import re
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = ROOT / "models/docutrust-qwen-lora"
CHUNKS = ROOT / "data/processed/DOC002_chunks.jsonl"
QUESTIONS = ROOT / "data/evaluation/doc002_qa_eval.jsonl"
OUT = ROOT / "data/evaluation"

TOP_K = 4
DEVICE = "cuda"

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def abstention(answer):
    s = (answer or "").strip().lower()
    return s.startswith("insufficient_evidence") or s.startswith("insufficient evidence")

def evaluate_model(model, tokenizer, questions, contexts, retrieved_pages, label):
    results = []
    model.eval()

    for i, item in enumerate(questions):
        question = item["question"]
        system = (
            "You are a strict document-grounded question answering system. "
            "Use ONLY the supplied document passages. Do not use outside knowledge. "
            "If evidence is insufficient, respond exactly: "
            "INSUFFICIENT_EVIDENCE: The supplied passages do not contain the answer. "
            "Otherwise answer concisely and cite every factual claim using [Page N]. "
            "Only cite pages present in the supplied passages."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content":
                f"Question: {question}\n\nDocument passages:\n{contexts[i]}"}
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        try:
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=220,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            answer = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            error = ""
        except Exception as exc:
            answer = ""
            error = str(exc)
            print("Generation error:", error)

        is_abstained = abstention(answer)
        cited = sorted(set(int(x) for x in re.findall(
            r"\[\s*Page\s+(\d+)", answer, re.I
        )))
        invalid = [p for p in cited if p not in retrieved_pages[i]]
        citation_valid = bool(cited) and not invalid and not is_abstained

        expected = set(int(x) for x in item.get("expected_pages", []))
        recall = (
            len(expected.intersection(retrieved_pages[i])) / len(expected)
            if expected else ""
        )

        results.append({
            "id": item["id"],
            "question": question,
            "is_answerable": item["is_answerable"],
            "expected_pages": ",".join(map(str, sorted(expected))),
            "retrieved_pages": ",".join(map(str, retrieved_pages[i])),
            "page_recall": round(recall, 3) if recall != "" else "",
            "answer": answer,
            "predicted_abstention": is_abstained,
            "citation_valid": citation_valid,
            "cited_pages": ",".join(map(str, cited)),
            "error": error
        })

        print(f"{label}: {i+1}/{len(questions)} | abstained={is_abstained}")

    path = OUT / f"doc002_{label.lower()}_results.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    successful = [r for r in results if not r["error"]]
    answerable = [r for r in successful if r["is_answerable"]]
    unanswerable = [r for r in successful if not r["is_answerable"]]
    correct = (
        sum(not r["predicted_abstention"] for r in answerable)
        + sum(r["predicted_abstention"] for r in unanswerable)
    )
    non_abstained = [r for r in successful if not r["predicted_abstention"]]

    metrics = {
        "model": label,
        "responses": len(successful),
        "failures": len(results) - len(successful),
        "answerability_accuracy": round(correct / len(successful), 3) if successful else 0,
        "answerable_non_abstentions": f"{sum(not r['predicted_abstention'] for r in answerable)}/{len(answerable)}",
        "unanswerable_correct_abstentions": f"{sum(r['predicted_abstention'] for r in unanswerable)}/{len(unanswerable)}",
        "citation_consistency": f"{sum(r['citation_valid'] for r in non_abstained)}/{len(non_abstained)}",
        "average_expected_page_recall": round(
            sum(r["page_recall"] for r in answerable if r["page_recall"] != "")
            / max(1, sum(r["page_recall"] != "" for r in answerable)), 3
        )
    }

    print("\n", label, "METRICS")
    for k, v in metrics.items():
        print(k, ":", v)
    print("Saved:", path)
    return metrics

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")

    print("Loading data...")
    chunks = load_jsonl(CHUNKS)
    questions = load_jsonl(QUESTIONS)
    print("Chunks:", len(chunks), "| Questions:", len(questions))

    print("Loading MiniLM retrieval model on CPU...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    embeddings = embedder.encode(
        [c["text"] for c in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    contexts, pages = [], []
    for item in questions:
        q = embedder.encode(
            [item["question"]],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")[0]

        scores = embeddings @ q
        indices = np.argsort(scores)[::-1][:TOP_K]
        selected = [chunks[int(j)] for j in indices]

        contexts.append("\n\n".join(
            f"[Page {c['page_number']} | {c.get('chunk_id', '')}]\n{c['text']}"
            for c in selected
        ))
        pages.append(sorted(set(int(c["page_number"]) for c in selected)))

    del embedder, embeddings
    gc.collect()

    print("\nLoading Qwen tokenizer and base model on GPU...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16
    ).to(DEVICE)
    base.config.use_cache = True

    base_metrics = evaluate_model(
        base, tokenizer, questions, contexts, pages, "base"
    )

    print("\nLoading trained LoRA adapter...")
    tuned = PeftModel.from_pretrained(base, str(ADAPTER))
    tuned.config.use_cache = True

    tuned_metrics = evaluate_model(
        tuned, tokenizer, questions, contexts, pages, "finetuned"
    )

    summary_path = OUT / "doc002_base_vs_finetuned_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {"base": base_metrics, "finetuned": tuned_metrics},
            f, indent=2
        )

    print("\n========== BASE vs FINE-TUNED ==========")
    for metric in [
        "answerability_accuracy",
        "answerable_non_abstentions",
        "unanswerable_correct_abstentions",
        "citation_consistency",
        "average_expected_page_recall"
    ]:
        print(
            metric,
            "| Base:", base_metrics[metric],
            "| Fine-tuned:", tuned_metrics[metric]
        )

    print("\nComparison saved:", summary_path)

if __name__ == "__main__":
    main()
