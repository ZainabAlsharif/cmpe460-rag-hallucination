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
| 3 | Qwen3-1.7B + LoRA | RAGBench | 0.874 | 0.535 | `configs/person3_zainab/qwen3_ragbench.yaml` | macro-F1 0.731, AUROC 0.886, threshold 0.47 (tuned on validation), 2740 test examples (finqa/techqa/emanual) |
| 3 | Qwen3-1.7B + LoRA | TRIVIA+ | 0.766 | 0.662 | `configs/person3_zainab/qwen3_triviaplus.yaml` | macro-F1 0.742, AUROC 0.827, threshold 0.62 (tuned on validation), 645 test examples |
| 3 | Llama-3.2-1B + LoRA | RAGBench | | | `configs/person3_zainab/llama_ragbench.yaml` | |
| 3 | Llama-3.2-1B + LoRA | TRIVIA+ | | | `configs/person3_zainab/llama_triviaplus.yaml` | |

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
| 3 | Train RAGBench (GPT-4 labels) → test TRIVIA+ (human labels), no further training | Qwen3: acc 0.643, F1 0.281, macro-F1 0.522, AUROC 0.650 (vs. 0.742 macro-F1 / 0.827 AUROC when trained on TRIVIA+ itself) |
| 3 | Qwen3 on TRIVIA+: trained on LLM-judge labels vs. human labels (`qwen3_triviaplus_llmlabels.yaml`) | |
| 3 | Zero-shot baselines (no fine-tuning): Qwen3 / Llama on RAGBench and TRIVIA+ | Qwen3: TRIVIA+ acc 0.640, F1 0.025, macro-F1 0.402, AUROC 0.540; RAGBench acc 0.875, F1 0.006, macro-F1 0.470, AUROC 0.671 (almost always answers "supported"). Llama: pending |
| 3 | (optional) LoRA rank 8 vs 32 | |
