# Person 3 (Zainab) — Small LLM judges fine-tuned with LoRA

**Research question:** Does a small fine-tuned LLM judge beat encoders, and does it transfer to new domains and human labels?

## Datasets

### 1. RAGBench (labels by GPT-4)
- Hugging Face: https://huggingface.co/datasets/rungalileo/ragbench · Paper: https://arxiv.org/abs/2407.11005
- Saved to: `data/person3_zainab/ragbench/<subset>/`
- Subsets used: `finqa`, `techqa`, `emanual` (edit `SUBSETS` in `download.py` to change).
  - ⚠️ `covidqa` is left out because TRIVIA+ also has COVID questions; `delucionqa` is left out (re-labelled copy of another dataset).
- Fields used: `question`, `documents` (list of passages → join into one context), `response`, `adherence_score` (True = supported → label 0, False = hallucinated → label 1).
- Sizes (train / val / test), checked on Kaggle 23 Nov:

  | Subset | Train | Val | Test | Supported / hallucinated (train) |
  |---|---|---|---|---|
  | finqa | 12,502 | 1,766 | 2,294 | 11,487 / 1,015 |
  | techqa | 1,192 | 304 | 314 | 746 / 446 |
  | emanual | 1,054 | 132 | 132 | 901 / 153 |

- ⚠️ **Imbalanced:** only ~11% hallucinated. Subsample finqa to ~4,000 rows, use class weights, and report F1 (not just accuracy).

### 2. TRIVIA+ (labels by humans)
- GitHub: https://github.com/amazon-science/hallucination-benchmark-trivialplus · Paper: https://arxiv.org/abs/2605.11330 (ACL 2026)
- Saved to: `data/person3_zainab/triviaplus/triviaplus_dataset.parquet` (one file, `split` column = train / valid / test)
- Licence: CC BY-NC-ND 4.0 (fine for coursework; never redistribute the data).
- Fields used: `article` (context), `question`, `answer` (LLM response), `response_level_label_binary` (0 = faithful, 1 = hallucinated; human majority vote).
- Checked on Kaggle 23 Nov:
  - Splits: train 2,263 · valid 316 · test 645
  - Hallucinated: 34% train · 39% valid · 35% test (much better balance than RAGBench)
  - Context length: median 1,840 characters, 90th percentile 30,598 → truncate to ~2,048 tokens
  - Sources: drop 1,339 · nq 674 · msmarco 521 + ms_marco 242 · trivia 309 · covid 139
  - Extra columns: `response_level_label` (0 faithful / 1 contradicted / −1 not mentioned) and `response_level_label_binary_ws_llm_aaj` (labels from an LLM-as-judge; agrees with humans on 75.3% of rows)

## Models
- Qwen3-1.7B — https://huggingface.co/Qwen/Qwen3-1.7B (debug first with https://huggingface.co/Qwen/Qwen3-0.6B)
- Llama-3.2-1B-Instruct — https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct (access requested 23 Nov; fallback https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)

Setup on T4: 4-bit QLoRA, gradient checkpointing, batch 1, grad-accum 16, fp16. Set `pad_token_id` for Llama.

## Extra experiments for the Discussion
1. **Transfer:** train on RAGBench (GPT-4 labels) → test on TRIVIA+ (human labels) without further training. No extra training run.
2. **Human vs. LLM-judge labels:** train Qwen3 on TRIVIA+ using `response_level_label_binary_ws_llm_aaj` instead of the human label, test on the human test labels. One extra run (`configs/person3_zainab/qwen3_triviaplus_llmlabels.yaml`).
3. *(Optional, if time)* LoRA rank 8 vs. 32.

## Download
```bash
python data/person3_zainab/download.py
```
