# DocuTrust AI — Integrated local app

This integration keeps the selected stack: Aya Expanse 8B through Ollama, MiniLM embeddings, FAISS retrieval, PyMuPDF extraction, and Streamlit.

## Prerequisites

- Windows 10/11
- Python 3.10–3.13 recommended for broad ML package compatibility. If your existing Python 3.14 environment already has the dependencies installed and working, you can try it first.
- Ollama for Windows
- Aya Expanse model: `ollama pull aya-expanse:8b`

## Setup (PowerShell, from the project folder)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
ollama pull aya-expanse:8b
```

If you already have a working `.venv` with PyTorch, Sentence Transformers, FAISS, and Ollama, do not recreate it; install only missing packages.

## Run

```powershell
python run.py
```

The launcher checks Ollama and the Aya model, then starts Streamlit.

## Usage

1. Open the local Streamlit URL shown in the terminal (normally http://localhost:8501).
2. Upload a text-based PDF.
3. Ask a question.
4. Expand Retrieved evidence to inspect the source chunks and physical PDF page numbers.

## Important limitations

- This version extracts selectable PDF text. It does not perform OCR on image-only scanned PDFs.
- The citation check verifies that cited pages were among retrieved pages. It does not prove that a citation supports its sentence.
- Aya is not fine-tuned in this integration. The existing pilot dataset is not sufficient to claim a fine-tuning improvement.
- Similarity scores are retrieval rankings, not confidence or accuracy percentages.
- Do not describe an answer as factually verified solely because the citation check passes.

## Files

- `app.py`: Streamlit interface and PDF QA pipeline.
- `run.py`: one-command startup and prerequisite checks.
- `src/evaluation/verify_answer.py`: reusable conservative citation checks.
- `requirements.txt`: app dependencies.
