# Hallucination Detection in RAG Systems

CMPE460 Deep Learning — Final Project (Group of 3)

We compare three ways to build a hallucination detector for Retrieval-Augmented Generation (RAG):

| Person | Approach | Datasets | Models |
|---|---|---|---|
| **Person 1 (Riahana)** | Response-level encoder classifiers | [HaluEval-QA](https://huggingface.co/datasets/pminervini/HaluEval), [FaithDial](https://huggingface.co/datasets/McGill-NLP/FaithDial) | [RoBERTa-base](https://huggingface.co/FacebookAI/roberta-base), [DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base) |
| **Person 2 (Abdulrahman)** | Span-level long-context token taggers | [RAGTruth](https://huggingface.co/datasets/wandb/RAGTruth-processed), [PsiloQA](https://huggingface.co/datasets/s-nlp/PsiloQA) | [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base), [Longformer-base-4096](https://huggingface.co/allenai/longformer-base-4096) |
| **Person 3 (Zainab)** | Small LLM judges fine-tuned with LoRA | [RAGBench](https://huggingface.co/datasets/rungalileo/ragbench), [DelucionQA](https://github.com/boschresearch/DelucionQA) | [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) |

## Repository layout

```
.
├── data/                       # one folder per person; raw data is downloaded here, NOT committed
│   ├── person1_riahana/        # README.md + download.py → HaluEval-QA, FaithDial
│   ├── person2_abdulrahman/    # README.md + download.py → RAGTruth, PsiloQA
│   └── person3_zainab/         # README.md + download.py → RAGBench, DelucionQA
├── src/
│   ├── common/                 # shared code (metrics, seeding) — used by everyone
│   ├── person1_riahana/        # Riahana's dataset classes + model wrappers
│   ├── person2_abdulrahman/    # Abdulrahman's token-classification code
│   └── person3_zainab/         # Zainab's LoRA code
├── configs/                    # one YAML per training run, one folder per person
├── results/results.md          # shared results table — fill in as runs finish
└── report/                     # LaTeX report (template460) — also on Overleaf
```

## Quick start (Colab / Kaggle)

```bash
git clone https://github.com/<OWNER>/cmpe460-rag-hallucination.git
cd cmpe460-rag-hallucination
pip install -r requirements.txt
python data/person3_zainab/download.py   # use your own folder: person1_riahana / person2_abdulrahman / person3_zainab
```

Each `download.py` saves both of that person's datasets into its own folder
(e.g. `data/person3_zainab/ragbench/`, `data/person3_zainab/delucionqa/`).

See [CONTRIBUTING.md](CONTRIBUTING.md) for the team workflow (branches, where to save checkpoints, what never to commit).

## Submission rules (from the project spec)

- Code is submitted as `[group id cmpe460 project].zip` — **no datasets, no model weights**. `data/**` contents and checkpoints are git-ignored so `git archive` gives a clean zip:
  `git archive -o cmpe460_project.zip HEAD`
- Code must use classes and separate files; notebooks are only launchers and are never submitted.
- Report: PDF built from `report/main.tex`. Presentation: PDF, ≤ 20 slides, ≤ 2 sentences per slide.
- Deadline: **06.12.2026 12:00**. One person submits for the whole group.
