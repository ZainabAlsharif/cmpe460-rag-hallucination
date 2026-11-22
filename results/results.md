# Results

Fill in each row as soon as a run finishes (test set). F1 = F1 of the *hallucinated* class.
Use `src/common/metrics.py → binary_metrics()` so everyone computes the numbers the same way.

## Main table

| Person | Model | Dataset | Acc | F1 | Config | Notes |
|---|---|---|---|---|---|---|
| 1 | RoBERTa-base | HaluEval-QA | | | `configs/person1_riahana/roberta_halueval.yaml` | |
| 1 | RoBERTa-base | FaithDial | | | `configs/person1_riahana/roberta_faithdial.yaml` | |
| 1 | DeBERTa-v3-base | HaluEval-QA | | | `configs/person1_riahana/deberta_halueval.yaml` | |
| 1 | DeBERTa-v3-base | FaithDial | | | `configs/person1_riahana/deberta_faithdial.yaml` | |
| 2 | ModernBERT-base | RAGTruth | | | `configs/person2_abdulrahman/modernbert_ragtruth.yaml` | example-level |
| 2 | ModernBERT-base | PsiloQA | | | `configs/person2_abdulrahman/modernbert_psiloqa.yaml` | example-level |
| 2 | Longformer-base-4096 | RAGTruth | | | `configs/person2_abdulrahman/longformer_ragtruth.yaml` | example-level |
| 2 | Longformer-base-4096 | PsiloQA | | | `configs/person2_abdulrahman/longformer_psiloqa.yaml` | example-level |
| 3 | Qwen3-1.7B + LoRA | RAGBench | | | `configs/person3_zainab/qwen3_ragbench.yaml` | |
| 3 | Qwen3-1.7B + LoRA | DelucionQA | | | `configs/person3_zainab/qwen3_delucionqa.yaml` | |
| 3 | Llama-3.2-1B + LoRA | RAGBench | | | `configs/person3_zainab/llama_ragbench.yaml` | |
| 3 | Llama-3.2-1B + LoRA | DelucionQA | | | `configs/person3_zainab/llama_delucionqa.yaml` | |

## Person 2 (Abdulrahman) — span-level scores

| Model | Dataset | Token F1 | Span F1 |
|---|---|---|---|
| ModernBERT-base | RAGTruth | | |
| ModernBERT-base | PsiloQA | | |
| Longformer-base-4096 | RAGTruth | | |
| Longformer-base-4096 | PsiloQA | | |

## Ablations

| Person | Experiment | Result |
|---|---|---|
| 1 | Answer-only input (no context) | |
| 1 | Length-only baseline | |
| 2 | Context truncated to 512 tokens | |
| 3 | Train RAGBench → test DelucionQA (no further training) | |
| 3 | LoRA rank 8 vs 32 | |
