# DocuTrust-AI
### An Evidence-Grounded AI Document Assistant for More Reliable Answers

**DocuTrust-AI** is an AI-powered document assistant designed to answer questions from uploaded PDF documents using retrieved evidence, page references, and evidence-aware response handling.

## 🚀 Live Demo
[Try DocuTrust-AI](https://docutrust-ai-djwjwyyk62nbhwxribvcyh.streamlit.app)

## 🎯 Problem
General-purpose AI systems can generate answers that sound convincing even when the supporting information is missing from a document. This creates a reliability challenge for document-based question answering.

## 💡 Our Approach
DocuTrust-AI combines:
- PDF text extraction using PyMuPDF.
- Semantic retrieval using Sentence Transformers.
- FAISS-based similarity search.
- Aya Expanse 8B through Ollama for answer generation.
- Evidence-aware responses with page references.
- Abstention handling for questions that cannot be supported by retrieved evidence.

## 🏗️ Technology Stack
| Component | Technology |
|---|---|
| Frontend | Streamlit |
| PDF processing | PyMuPDF |
| Embeddings | all-MiniLM-L6-v2 |
| Vector search | FAISS |
| Language model | Aya Expanse 8B |
| Model runtime | Ollama |
| Programming language | Python |

## 📊 Evaluation
One 50-question hybrid balanced evaluation run produced:

- Answerability accuracy: **84%**
- Correct abstentions on unanswerable questions: **19/20**
- Expected page recall: **84.4%**
- Citation consistency: **24/24 checked citations**

A separate stricter verified evaluation run produced 72% answerability accuracy and 18/20 correct abstentions.

These are experimental results on our evaluation set, not guarantees of general performance. Citation consistency is measured only on the citations checked in each run.

## 📚 Dataset
Our public pilot dataset contains four synthetic/demo QA examples:
- 3 answerable examples
- 1 unanswerable example

[View Curated QA Dataset](https://huggingface.co/datasets/GARVITb/docutrust-curated-qa)

## 🤖 Model Artifacts
An experimental Qwen LoRA adapter is available separately:

[View Hugging Face Model](https://huggingface.co/GARVITb/docutrust-qwen-lora)

This is a supplementary experimental artifact and is not the Aya Expanse model used by the main application.

## 👥 Team
**CortexLabs**
- Garvit Bohara
- Suhail

## ⚠️ Limitations
The current evaluation is limited in size and scope. Retrieval metrics and citation checks do not establish full semantic correctness. Larger datasets, broader testing, and further human evaluation are needed.

## 🔮 Future Work
- Expand multilingual document support.
- Improve retrieval and evidence verification.
- Build larger and more diverse evaluation datasets.
- Conduct more extensive hallucination and answer-quality testing.

## 📄 License
Please refer to the repository's license information before reusing this project.
