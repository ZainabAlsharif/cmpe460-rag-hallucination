# Person 2 (Abdulrahman) — Span-level long-context token taggers

**Research question:** Can we locate *which words* in an answer are unsupported when the full context is visible?

## Datasets

### 1. RAGTruth (processed)
- Hugging Face: https://huggingface.co/datasets/wandb/RAGTruth-processed
- Original repo: https://github.com/ParticleMedia/RAGTruth · Paper: https://arxiv.org/abs/2401.00396
- Saved to: `data/person2_abdulrahman/ragtruth/`
- ~18k LLM responses (QA, summarization, data-to-text) with human word-level hallucination spans.
- Span types: Evident/Subtle Conflict, Evident/Subtle Baseless Info (useful for the optional 3-class extension).

### 2. PsiloQA (English portion)
- Hugging Face: https://huggingface.co/datasets/s-nlp/PsiloQA
- Original repo: https://github.com/s-nlp/PsiloQA · Paper: https://arxiv.org/abs/2510.04849
- Saved to: `data/person2_abdulrahman/psiloqa/` (the download script keeps English rows only if it finds a language column — check the printed message)
- Wikipedia contexts, span-level labels annotated by GPT-4o (contrast with RAGTruth's human labels).

## Labels
Spans are given as character offsets in the answer → map them to tokens with the tokenizer's `offset_mapping`.
Label answer tokens 0/1; set context and question tokens to `-100` so the loss ignores them (LettuceDetect recipe).
**Check alignment on ~5 examples** by printing the tokens labelled 1.

## Models
- ModernBERT-base — https://huggingface.co/answerdotai/ModernBERT-base (use `attn_implementation="sdpa"` on T4)
- Longformer-base-4096 — https://huggingface.co/allenai/longformer-base-4096

## Reference
LettuceDetect — paper https://arxiv.org/abs/2502.17125 · code https://github.com/KRLabsOrg/LettuceDetect

## Ablation for the Discussion
Truncate the context to 512 tokens vs. full length.

## Download
```bash
python data/person2_abdulrahman/download.py
```
