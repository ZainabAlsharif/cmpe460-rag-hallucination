# Person 3 (Zainab) — Small LLM judges fine-tuned with LoRA

**Research question:** Does a small fine-tuned LLM judge beat encoders, and does it transfer across domains and label sources?

## Datasets

### 1. RAGBench
- Hugging Face: https://huggingface.co/datasets/rungalileo/ragbench · Paper: https://arxiv.org/abs/2407.11005
- Saved to: `data/person3_zainab/ragbench/<subset>/`
- Subsets downloaded by default: `covidqa`, `finqa`, `techqa`, `emanual` (edit `SUBSETS` in `download.py` to change).
  ⚠️ Do **not** use the `delucionqa` subset — it overlaps with dataset 2.
- Fields used: `question`, `documents` (list of passages → join into one context), `response`, `adherence_score` (label; GPT-4 generated).
- Aim for ~8k training rows in total; max input length ~2048 tokens. Check the class balance per subset.

### 2. DelucionQA
- GitHub: https://github.com/boschresearch/DelucionQA · Paper: https://arxiv.org/abs/2312.05200
- Saved to: `data/person3_zainab/delucionqa/{train,dev,test}.csv`
- Jeep 2023 Gladiator car manual QA, **human-annotated**.
- Labels are per sentence → an answer is hallucinated if any of its sentences is. Open the CSV first and confirm the column names.

## Models
- Qwen3-1.7B — https://huggingface.co/Qwen/Qwen3-1.7B (start with Qwen3-0.6B to debug: https://huggingface.co/Qwen/Qwen3-0.6B)
- Llama-3.2-1B-Instruct — https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct (gated: request access; fallback https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)

Setup on T4: 4-bit QLoRA, gradient checkpointing, batch 1, grad-accum 16, fp16. Set `pad_token_id` for Llama.

## Ablations for the Discussion
1. Train on RAGBench → test on DelucionQA without further training (GPT-4 labels → human labels).
2. LoRA rank 8 vs. 32.

## Download
```bash
python data/person3_zainab/download.py
```
