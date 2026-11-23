# Person 3 (Zainab) — LLM judges with LoRA: how to run

Everything runs from the **repository root** on a GPU machine (Kaggle: *Settings → Accelerator → GPU T4 x2*,
*Settings → Internet on*). The code uses one GPU; the second T4 is simply ignored.

## 0. One-time setup (every new Kaggle session)

```bash
pip install -q -U "transformers>=4.51,<5" "peft>=0.11" "bitsandbytes>=0.43" accelerate
python data/person3_zainab/download.py          # RAGBench (finqa, techqa, emanual) + TRIVIA+
```

## 1. Smoke test (~10 minutes) — always run this first

```bash
python -m src.person3_zainab.train --config configs/person3_zainab/debug_qwen3_0.6b.yaml
```
Uses Qwen3-0.6B, 200 training rows and 100 test rows. Results go to `results/person3_zainab/debug/`
(not mixed with the real results). If this finishes and prints `TEST acc=... f1=...`, the pipeline works.

## 2. The four main runs (the group results table)

| # | Command | Rough time on a T4 |
|---|---|---|
| 1 | `python -m src.person3_zainab.train --config configs/person3_zainab/qwen3_ragbench.yaml` | 3–5 h |
| 2 | `python -m src.person3_zainab.train --config configs/person3_zainab/qwen3_triviaplus.yaml` | 3–5 h |
| 3 | `python -m src.person3_zainab.train --config configs/person3_zainab/llama_ragbench.yaml` | 2–4 h |
| 4 | `python -m src.person3_zainab.train --config configs/person3_zainab/llama_triviaplus.yaml` | 2–4 h |

How the times were estimated: the smoke test measured ~1.25 training examples/s for Qwen3-0.6B at
1,024 tokens; a 1.7B model at 1,536 tokens is roughly 3–4× slower. **Check the progress bar's time
estimate in the first minutes.** If a run would not fit in one Kaggle session (12 h), add
`--max-train-samples 3000` (or lower).

**Long runs on Kaggle:** use *Save Version → Save & Run All (Commit)* so the notebook runs in the
background (up to 12 h) even if you close the browser; results appear in the version's *Output* tab.

If Llama access is still pending, set `model_name: HuggingFaceTB/SmolLM2-1.7B-Instruct` in configs 3–4.

Each run writes `results/person3_zainab/<run_name>.json` (+ `_predictions.csv`) and saves the trained
LoRA adapter and the tuned threshold to `outputs/<run_name>/`. **Download the `outputs/` folder**
(Output panel) if you want to run the transfer experiment in a later session.

## 3. Extra experiments (for the Discussion)

**a) Transfer: RAGBench (GPT-4 labels) → TRIVIA+ (human labels), no retraining** — needs run 1 (and 3):
```bash
python -m src.person3_zainab.evaluate --run-dir outputs/p3_qwen3_ragbench \
    --data-config configs/person3_zainab/qwen3_triviaplus.yaml --name p3_qwen3_transfer_ragbench_to_triviaplus
python -m src.person3_zainab.evaluate --run-dir outputs/p3_llama_ragbench \
    --data-config configs/person3_zainab/llama_triviaplus.yaml --name p3_llama_transfer_ragbench_to_triviaplus
```

**b) LLM-judge labels vs. human labels on TRIVIA+** — one extra training run, compare with run 2:
```bash
python -m src.person3_zainab.train --config configs/person3_zainab/qwen3_triviaplus_llmlabels.yaml
```

**c) Zero-shot baselines (no training, ~5–15 min each)** — how good is the judge *before* fine-tuning?
```bash
python -m src.person3_zainab.evaluate --zero-shot --data-config configs/person3_zainab/qwen3_ragbench.yaml --name p3_qwen3_zeroshot_ragbench
python -m src.person3_zainab.evaluate --zero-shot --data-config configs/person3_zainab/qwen3_triviaplus.yaml --name p3_qwen3_zeroshot_triviaplus
python -m src.person3_zainab.evaluate --zero-shot --data-config configs/person3_zainab/llama_ragbench.yaml --name p3_llama_zeroshot_ragbench
python -m src.person3_zainab.evaluate --zero-shot --data-config configs/person3_zainab/llama_triviaplus.yaml --name p3_llama_zeroshot_triviaplus
```

**d) Optional: LoRA rank ablation**
```bash
python -m src.person3_zainab.train --config configs/person3_zainab/qwen3_triviaplus.yaml --lora-r 8
python -m src.person3_zainab.train --config configs/person3_zainab/qwen3_triviaplus.yaml --lora-r 32
```

## 4. Tables and figures for the report

```bash
python -m src.person3_zainab.summarize
```
Writes `results/person3_zainab/summary.md`, `report/tables/person3_results.tex` (`\input{tables/person3_results}`
in the report) and four figures in `report/figures/person3/`: F1 per run, confusion matrices, F1 per
source/domain and ROC curves (one panel per test set). Commit `results/` and `report/` afterwards.

The **architecture diagram** for the Methodology section is already in the repo
(`report/figures/person3/architecture.pdf`); regenerate it with `python -m src.person3_zainab.figures`.

```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=\linewidth]{figures/person3/architecture.pdf}
    \caption{The LLM judge: a 4-bit decoder LLM with LoRA adapters, read out through the " Yes"/" No" tokens.}
    \label{fig:p3-architecture}
\end{figure}
```

## What each file does

| File | Class / function | Purpose |
|---|---|---|
| `config.py` | `ExperimentConfig` | Reads a YAML config; validates it; defaults for every setting |
| `data.py` | `RAGBenchLoader`, `TriviaPlusLoader` | Load both datasets into one format: context, question, response, label (1 = hallucinated), source |
| `prompt.py` | `PromptEncoder` | Builds the judge prompt as token ids; cuts only the context when too long |
| `model.py` | `JudgeModelFactory`, `Verbalizer` | 4-bit NF4 base model + LoRA adapters; the class scores are the model's own next-token scores for " Yes" (supported) / " No" (hallucinated) |
| `trainer.py` | `WeightedLossTrainer` | Per-example class-weighted cross-entropy (RAGBench is ~11% hallucinated) |
| `evaluation.py` | `Predictor`, `compute_report` | Batched inference; accuracy, P/R/F1 (hallucinated class), macro-F1, AUROC, confusion matrix, per-source scores |
| `train.py` | `Experiment` | Full run: data → encode → train (best epoch by validation macro-F1) → decision threshold tuned on validation → test → save |
| `evaluate.py` | `JudgeEvaluation` | Test a trained adapter on another dataset (transfer), or the untrained model (zero-shot) |
| `summarize.py` | `ResultsSummary` | Markdown + LaTeX tables and figures (F1, confusion matrices, per-source F1, ROC curves) |
| `figures.py` | `ArchitectureDiagram` | Architecture diagram for the Methodology section |

Local checks that need no GPU: `python tests/test_person3.py`.

## Design decisions (worth explaining in the Methodology / Discussion)

* **Yes/No verbaliser instead of a new classification layer.** A randomly initialised head on top of
  the LLM learned almost nothing in the smoke test (all predictions within 0.83–0.85). Reading the
  pretrained LM head's scores for " Yes" / " No" fixed this and also makes a zero-shot baseline possible.
* **Model selection by macro-F1, not F1 of the hallucinated class.** The latter is maximised by calling
  *everything* hallucinated (that epoch scored F1 = 0.55 while being useless).
* **Threshold tuned on validation.** Fine-tuned LLMs are often mis-calibrated; the cut-off on
  P(hallucinated) is chosen on the validation split and applied unchanged to the test split.
  Results at the default 0.5 are also stored (`test_at_0.5`).
* **Class weights** for the imbalanced RAGBench (~11% hallucinated), applied per example.

Smoke-test sanity check (Qwen3-0.6B, 800 TRIVIA+ training rows, 100 test rows, 23 Nov):
fine-tuned macro-F1 0.66 / AUROC 0.76 vs. zero-shot macro-F1 0.42 / AUROC 0.53.

## Why some numbers will not be perfect

No hallucination detector reaches 100% — published detectors score far lower on these benchmarks
(e.g. LettuceDetect reports 76.8% F1 on RAGTruth). A test score near 100% would suggest a bug such as
test data leaking into training. The code is built so that the numbers it reports are trustworthy:
official test splits are never used for training or model selection, the best epoch is chosen on the
validation split only, and F1 of the hallucinated class is reported alongside accuracy so class
imbalance cannot hide a weak model.
