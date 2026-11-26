# Person 1 (Riahana): response-level encoder classifiers

RoBERTa-base and DeBERTa-v3-base decide whether a whole response is hallucinated, on HaluEval-QA and FaithDial.

Research question: does a short-context cross-encoder learn real grounding, or shortcuts?

All commands are run from the repository root.

## 1. Setup

```bash
pip install -r requirements.txt
python data/person1_riahana/download.py
```

## 2. Smoke test

```bash
python -m src.person1_riahana.train --config configs/person1_riahana/roberta_halueval.yaml \
    --max-train-examples 480 --max-eval-examples 200 --epochs 1
```

Debug results go to `results/person1_riahana/debug/`.

## 3. All experiments

```bash
python -m src.person1_riahana.run_all                # GPU with enough memory (for example a T4)
python -m src.person1_riahana.run_all --low-memory   # 4 GB GPU
```

Finished runs are skipped, so the command can be restarted after an interruption.

The reported numbers are the mean and standard deviation over three training seeds (42, 13 and 2024):

```bash
python -m src.person1_riahana.run_all --low-memory --seeds 13 2024
python -m src.person1_riahana.summarize     # tables and figures
python -m src.person1_riahana.analysis      # significance tests and error analysis
```

A seed changes the classifier head, the batch order and dropout. The data split and the context shuffling use
`split_seed` and stay the same in every repetition.

| Experiment | Question it answers |
|---|---|
| 4 main runs (2 models x 2 datasets) | How well do the encoders detect hallucinations? |
| Length-only and overlap baselines | Can the datasets be solved from surface features? |
| `--input-mode answer_only` (4 runs) | How much can be learned without any context? |
| `--shuffle-context` evaluation | Does a trained model still use the context it is given? |
| Train on one dataset, test on the other | Does the skill transfer? |

`--low-memory` keeps the effective batch size at 16 and uses 8-bit AdamW for both models.
RoBERTa runs with batch size 8 x 2 accumulation steps (fp16). DeBERTa-v3 uses bfloat16 mixed precision with
batch size 2 x 8 steps on HaluEval and 4 x 4 steps on FaithDial.

## Files

| File | Content |
|---|---|
| `config.py` | `ExperimentConfig`: reads a YAML file, command line overrides |
| `datasets.py` | `HaluEvalQALoader`, `FaithDialLoader`, context shuffling |
| `model.py` | `PairEncoder` (tokenisation) and the classifier |
| `train.py` | `Experiment`: train, choose the best epoch on validation macro-F1, test, save |
| `evaluate.py` | test a trained model on a wrong context or on the other dataset |
| `baselines.py` | logistic regression on length or word-overlap features |
| `evaluation.py` | prediction and metrics (uses `src/common/metrics.py`) |
| `run_all.py` | runs every experiment |
| `summarize.py` | tables and figures over all seeds |
| `analysis.py` | bootstrap and McNemar tests, context sensitivity, length-matched subset, error slices |

## Why the scores are high: not train/test leakage

Accuracy close to 100% is a reason to look for leakage, so it was checked
(`analysis.py`, section 5, written to `results/person1_riahana/analysis.md`).

* **HaluEval-QA:** no knowledge passage, context or (context, response) pair of the test set occurs in the
  training set. Short answer strings such as names do repeat across different passages (517 of 3,994 test
  examples).
* **FaithDial:** no (context, response) pair or response string of the test set occurs in the training set.
  1,605 of 5,275 test examples use a knowledge sentence that is also in the training data, but accuracy on the
  other 3,670 examples is nearly the same.
* **The dataset itself gives the answer away.** On HaluEval-QA a logistic regression on two word-overlap
  features reaches 98.4% and one on response length 94.4%, without any neural network. The encoders reach
  97.8% from the response alone.

The high scores therefore show that these test sets can be solved with surface cues. They are not evidence
that the models check facts, and the models do not transfer to the other dataset.

## Data decisions

* HaluEval-QA has one split only. It is divided 70/10/20 **by knowledge passage**, so the two answers of a
  question never end up in different splits.
* FaithDial keeps the official splits. Only turns whose original response is annotated as a hallucination are
  used: the original response is label 1, the edited response is label 0. Validation and test turns are annotated
  more than once, so identical examples are removed. Some knowledge sentences occur in both the training and the
  test split (620 of 1,690 test sentences), because this is how the official splits are built.
