# Person 1: analysis of the test predictions

## 1. Are the differences real?

Accuracy is averaged over the seeds. The interval comes from a paired bootstrap that resamples groups of related examples (same passage or same dialogue turn); it describes the uncertainty from the choice of test examples, not from retraining. McNemar p-values are given for each seed pair separately; that test treats the examples as independent. No correction for multiple comparisons is applied.

| Dataset | Comparison (A vs B) | Acc A | Acc B | A - B (points) | 95% interval | McNemar p per seed |
|---|---|---|---|---|---|---|
| HaluEval-QA | DeBERTa vs RoBERTa | 99.32 | 99.42 | -0.10 | [-0.23, +0.03] | 0.039, 0.38, 1 |
| HaluEval-QA | RoBERTa-base: full input vs response only | 99.42 | 97.76 | +1.65 | [+1.26, +2.05] | 3.1e-20, 3.5e-14, 2.4e-16 |
| HaluEval-QA | DeBERTa-v3-base: full input vs response only | 99.32 | 97.75 | +1.57 | [+1.19, +1.95] | 9.7e-17, 2.5e-15, 9.8e-16 |
| FaithDial | DeBERTa vs RoBERTa | 96.41 | 95.81 | +0.60 | [+0.23, +0.97] | 9.7e-05, 0.09, 0.1 |
| FaithDial | RoBERTa-base: full input vs response only | 95.81 | 93.66 | +2.15 | [+1.70, +2.61] | 2.9e-11, 2.7e-17, 7.1e-11 |
| FaithDial | DeBERTa-v3-base: full input vs response only | 96.41 | 93.48 | +2.93 | [+2.41, +3.46] | 2.7e-21, 1.3e-15, 2.7e-25 |

## 2. Does the model react to the context?

Share of grounded responses (label 0) that are called hallucinated. Mean ± standard deviation over the seeds.

| Dataset | Model | Right context | Wrong context |
|---|---|---|---|
| HaluEval-QA | RoBERTa-base | 0.1 ± 0.0% | 91.3 ± 1.3% |
| HaluEval-QA | DeBERTa-v3-base | 0.4 ± 0.3% | 91.9 ± 1.0% |
| FaithDial | RoBERTa-base | 3.2 ± 0.3% | 75.8 ± 4.4% |
| FaithDial | DeBERTa-v3-base | 2.6 ± 0.3% | 88.0 ± 1.2% |

## 3. Is length the only shortcut?

HaluEval-QA test responses with 4 to 6 words: 626 examples, 419 of them hallucinated. Balanced accuracy is the mean of the two per-class accuracies.

| Model | Input | Accuracy | Balanced accuracy |
|---|---|---|---|
| RoBERTa-base | context + response | 98.9 ± 0.2% | 99.1 ± 0.1% |
| RoBERTa-base | response only | 96.9 ± 0.2% | 97.2 ± 0.2% |
| DeBERTa-v3-base | context + response | 98.8 ± 0.1% | 98.8 ± 0.2% |
| DeBERTa-v3-base | response only | 96.8 ± 0.1% | 97.0 ± 0.1% |
| Baseline: length | surface features | 81.3 ± 0.0% | 77.8 ± 0.0% |
| Baseline: overlap | surface features | 97.3 ± 0.0% | 98.0 ± 0.0% |

## 4. Which examples go wrong?

Error rate of the full-input models, averaged over the seeds.

### HaluEval-QA

| Slice | Examples | RoBERTa-base | DeBERTa-v3-base |
|---|---|---|---|
| All | 3994 | 0.58% | 0.68% |
| Grounded (error = false alarm) | 1997 | 0.12% | 0.37% |
| Hallucinated (error = miss) | 1997 | 1.05% | 1.00% |
| 1-3 words | 1832 | 0.75% | 0.78% |
| 4-10 words | 1265 | 0.69% | 0.92% |
| 11-20 words | 732 | 0.09% | 0.18% |
| 21+ words | 165 | 0.20% | 0.00% |
| Grounded, copied word for word from the knowledge | 1912 | 0.12% | 0.38% |
| Grounded, not copied word for word | 85 | 0.00% | 0.00% |
| Hallucinated, copied word for word from the knowledge | 15 | 75.56% | 73.33% |
| Hallucinated, not copied word for word | 1982 | 0.49% | 0.45% |

3943 examples are right in all 6 runs; 16 are wrong in at least 5 runs (listed in `hard_examples_halueval_qa.csv`).

### FaithDial

| Slice | Examples | RoBERTa-base | DeBERTa-v3-base |
|---|---|---|---|
| All | 5275 | 4.19% | 3.59% |
| Grounded (error = false alarm) | 2668 | 3.17% | 2.62% |
| Hallucinated (error = miss) | 2607 | 5.23% | 4.58% |
| 1-3 words | 3 | 11.11% | 0.00% |
| 4-10 words | 373 | 7.69% | 5.99% |
| 11-20 words | 2835 | 4.60% | 3.97% |
| 21+ words | 2064 | 2.99% | 2.63% |
| Grounded, not copied word for word | 2668 | 3.17% | 2.62% |
| Hallucinated, copied word for word from the knowledge | 2 | 0.00% | 33.33% |
| Hallucinated, not copied word for word | 2605 | 5.23% | 4.56% |

4886 examples are right in all 6 runs; 114 are wrong in at least 5 runs (listed in `hard_examples_faithdial.csv`).
