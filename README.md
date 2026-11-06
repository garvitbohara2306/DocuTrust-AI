# DocuTrust AI

**Grounded AI Document Assistant with Fewer Hallucinations**

**Team:** CortexLabs
**Team Members:** Garvit Bohara and Suhail
**Track:** Fine-tuning and Data Curation

---

## 1. Problem Statement

Large Language Models (LLMs) can generate incorrect or misleading information, commonly known as hallucinations.

When users ask questions about documents, AI systems may sometimes provide answers that are not supported by the actual document content. This creates problems for reliability, accuracy and trust.

## 2. Our Solution

DocuTrust AI aims to build an intelligent document assistant that provides accurate, context-grounded answers from uploaded documents.

The system will focus on:

- Preparing high-quality, curated question-answer datasets.
- Improving model responses through fine-tuning.
- Retrieving relevant information from uploaded documents.
- Providing answers supported by document references.
- Evaluating hallucinations and answer quality.

## 3. Proposed System Workflow

1. User uploads a document.
2. System extracts text and page information.
3. Relevant document sections are retrieved.
4. The AI model generates an answer using retrieved context.
5. The application provides supporting document references.
6. Evaluation compares answer quality and groundedness.

## 4. AI / ML Pipeline

Our planned pipeline consists of:

**Stage 1 — Data Collection**
- Collect suitable source documents.
- Extract useful information.
- Prepare question-answer examples.

**Stage 2 — Data Curation**
- Remove duplicate and low-quality examples.
- Verify answer correctness.
- Prepare structured training data.

**Stage 3 — Model Fine-tuning**
- Select an eligible open-weights base model.
- Fine-tune using curated examples.
- Compare the original and improved models.

**Stage 4 — Evaluation**
- Evaluate answer correctness.
- Measure groundedness.
- Identify unsupported answers and hallucinations.
- Compare results before and after fine-tuning.

## 5. Technology Stack

| Component | Planned Technology |
|---|---|
| Programming | Python |
| AI Models | Eligible Cohere Labs open-weights model |
| Data processing | CuratorKIT |
| Fine-tuning | AlignTune |
| Evaluation | AuditKIT |
| Frontend | Streamlit |
| Document extraction | To be finalized |
| Version control | Git and GitHub |

> Tool compatibility and the final implementation will be verified before integration.

## 6. Project Structure

```
DocuTrust-AI/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│
├── src/
│   ├── document_processing/
│   ├── retrieval/
│   ├── model/
│   └── evaluation/
│
├── frontend/
├── configs/
├── notebooks/
├── outputs/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 7. Team Responsibilities

**Garvit Bohara — AI / ML Developer**
- Dataset preparation and curation.
- Base model experimentation.
- Fine-tuning pipeline.
- Model evaluation.
- Hallucination analysis.

**Suhail — Backend / Application Developer**
- PDF upload functionality.
- Document text extraction.
- Retrieval pipeline.
- Streamlit application.
- Source references and answer presentation.

**Joint Responsibilities**
- Model and application integration.
- Testing and debugging.
- Final demonstration.
- Hackathon documentation.

## 8. Expected Outcomes

- A working document-based AI assistant.
- Answers grounded in uploaded documents.
- Supporting page references.
- A curated training dataset.
- Experimental comparison of the original and fine-tuned models.
- Evaluation of answer quality and hallucinations.

## 9. Development Status

**Completed**
- [x] Initial project setup
- [x] GitHub repository created
- [x] Development environment initialized
- [x] Project folder structure created
- [x] Initial Git commit pushed

**Planned**
- [ ] Dataset preparation
- [ ] Model experimentation
- [ ] Fine-tuning
- [ ] Evaluation
- [ ] Application integration
- [ ] Final demonstration

---

*Team CortexLabs | DocuTrust AI*