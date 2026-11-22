# Person 1 (Riahana) — Response-level encoder classifiers

**Research question:** Does a short-context cross-encoder learn real grounding, or shortcuts?

## Datasets

### 1. HaluEval — QA subset
- Hugging Face: https://huggingface.co/datasets/pminervini/HaluEval (config `qa`)
- Original repo: https://github.com/RUCAIBox/HaluEval
- Saved to: `data/person1_riahana/halueval_qa/`
- Fields: `knowledge`, `question`, `right_answer`, `hallucinated_answer` (10,000 rows, one split called `data`)
- **Labels:** turn every row into two examples → (`right_answer`, 0) and (`hallucinated_answer`, 1) = 20k balanced examples.
- ⚠️ **Split by question**, not by example: both answers of a question must be in the same split, otherwise test data leaks into training.

### 2. FaithDial
- Hugging Face: https://huggingface.co/datasets/McGill-NLP/FaithDial
- Original repo: https://github.com/McGill-NLP/FaithDial
- Saved to: `data/person1_riahana/faithdial/`
- Fields: `history`, `knowledge`, `response`, `original_response`, `BEGIN`, `VRM`
- **Labels (FaithCritic recipe):** `original_response` whose `BEGIN` list contains `"Hallucination"` → 1; the edited `response` → 0. Use the provided train / validation / test splits.

## Models
- RoBERTa-base — https://huggingface.co/FacebookAI/roberta-base
- DeBERTa-v3-base — https://huggingface.co/microsoft/deberta-v3-base (if loss becomes NaN in fp16, train in fp32)

## Ablation for the Discussion
Train with the answer only (no knowledge/context) and add a length-only baseline → shows whether the model really reads the context.

## Download
```bash
python data/person1_riahana/download.py
```
