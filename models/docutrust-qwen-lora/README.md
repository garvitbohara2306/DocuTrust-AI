---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
---

# DocuTrust AI – Qwen LoRA Adapter

## Model Overview

DocuTrust AI is an experimental document-grounded question-answering project designed to generate answers using information retrieved from source documents.

This repository contains a **LoRA fine-tuned adapter** built on top of `Qwen/Qwen2.5-0.5B-Instruct`.

The adapter is an experimental component of the larger DocuTrust AI project. It is not a standalone language model.

## Model Details

| Field | Details |
|---|---|
| Model name | DocuTrust AI Qwen LoRA |
| Base model | Qwen/Qwen2.5-0.5B-Instruct |
| Fine-tuning method | LoRA / PEFT |
| Target modules | q_proj, v_proj |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Training epochs | 3 |
| Learning rate | 2e-4 |
| Maximum sequence length | 512 |
| Precision | FP16 |
| Hardware | NVIDIA RTX 3050 Laptop GPU (6 GB VRAM) |

## Intended Use

The adapter is intended for experimentation with:

- Document-grounded question answering.
- Fine-tuning small language models.
- Reducing unsupported answers in document QA workflows.
- Research into retrieval-augmented generation and hallucination evaluation.

It should be used with relevant document context retrieved by the DocuTrust AI retrieval pipeline.

## Training Data

The training data was prepared using content from:

**Process & Observations on Accessibility of PDF Documents on Indian Government Websites – Standard Operating Procedure, Version 2.0, August 2020.**

Dataset preparation:

| Dataset | Examples |
|---|---:|
| Training | 62 |
| Validation | 16 |
| Holdout | 5 |
| Total | 83 |

**Data quality note:** These examples were source-audited, but human approval of every training example has not been completed. They should not be described as fully human-verified.

## Training Results

Training loss across three epochs:

| Epoch | Training Loss | Validation Loss |
|---|---:|---:|
| 1 | 1.0455 | 1.0191 |
| 2 | 0.8713 | 1.0048 |
| 3 | 0.7857 | 1.0095 |

Training loss decreased across epochs. Validation loss improved initially and then increased slightly. These results alone do not establish improved answer quality.

## Evaluation and Limitations

The model was evaluated on a 50-question evaluation set containing:

- 30 answerable questions.
- 20 unanswerable questions.

The heuristic evaluation found:

| Metric | Base Model | Fine-tuned Model |
|---|---:|---:|
| Answerable questions answered (non-abstention) | 28/30 | 27/30 |
| Correct abstentions on unanswerable questions | 4/20 | 4/20 |
| Combined heuristic score | 0.64 | 0.62 |

These are heuristic evaluation results, not human-verified accuracy scores. The experiment does not demonstrate an improvement over the base model.

The retrieval pipeline was evaluated separately. Hybrid retrieval achieved **86.7% Page Hit@4**, compared with **70%** for dense retrieval on 30 answerable questions. This measures retrieval performance, not model answer accuracy.

Known limitations:

- The training dataset is small.
- Human approval of training examples is pending.
- Citation consistency remains weak.
- The model may generate unsupported answers.
- The model has not been validated for production use.
- Retrieval page matching does not guarantee that a generated claim is supported by its cited page.

## Usage

The adapter requires the original base model.

Example loading code:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_path = "./models/docutrust-qwen-lora"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    adapter_path
)

model.eval()
```

This example loads the adapter. A complete document QA workflow also requires document processing, retrieval, and answer verification.

## Ethical Considerations

This model is experimental and may produce inaccurate or unsupported information.

Users should verify generated answers against the original source document. It should not be used as the sole basis for legal, medical, financial, accessibility-compliance, or other high-impact decisions.

## Development Team

**Project:** DocuTrust AI
**Team:** CortexLabs
**Fine-tuning and dataset experiments:** Garvit Bohara

## Acknowledgements

- Qwen team for the Qwen2.5 base model.
- Hugging Face Transformers and PEFT libraries.
- The authors of the source accessibility SOP.

## License

The adapter's redistribution terms have not yet been independently established. Refer to the base model's license and applicable source-document terms before redistributing this adapter or its training data.
