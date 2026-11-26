# Person 1 results (test sets)

F1 is the F1 of the hallucinated class. Neural models: mean ± standard deviation over the training seeds (column Runs). The baselines are deterministic.

| Dataset | Model | Setting | Runs | Acc | F1 | Macro-F1 | AUROC |
|---|---|---|---|---|---|---|---|
| HaluEval-QA | RoBERTa-base | Context + response | 3 | 0.994 ± 0.000 | 0.994 ± 0.000 | 0.994 ± 0.000 | 0.998 ± 0.000 |
| HaluEval-QA | DeBERTa-v3-base | Context + response | 3 | 0.993 ± 0.001 | 0.993 ± 0.001 | 0.993 ± 0.001 | 0.998 ± 0.001 |
| HaluEval-QA | RoBERTa-base | Response only | 3 | 0.978 ± 0.001 | 0.977 ± 0.001 | 0.978 ± 0.001 | 0.989 ± 0.001 |
| HaluEval-QA | DeBERTa-v3-base | Response only | 3 | 0.977 ± 0.001 | 0.977 ± 0.001 | 0.977 ± 0.001 | 0.989 ± 0.000 |
| HaluEval-QA | RoBERTa-base | Wrong context | 3 | 0.544 ± 0.007 | 0.687 ± 0.003 | 0.423 ± 0.013 | 0.981 ± 0.004 |
| HaluEval-QA | DeBERTa-v3-base | Wrong context | 3 | 0.540 ± 0.005 | 0.685 ± 0.002 | 0.417 ± 0.009 | 0.883 ± 0.116 |
| HaluEval-QA | RoBERTa-base | Trained on the other dataset | 3 | 0.314 ± 0.019 | 0.458 ± 0.032 | 0.260 ± 0.010 | 0.368 ± 0.013 |
| HaluEval-QA | DeBERTa-v3-base | Trained on the other dataset | 3 | 0.481 ± 0.063 | 0.558 ± 0.013 | 0.455 ± 0.098 | 0.506 ± 0.052 |
| HaluEval-QA | Baseline: length | Surface features only | 1 | 0.944 | 0.943 | 0.944 | 0.972 |
| HaluEval-QA | Baseline: overlap | Surface features only | 1 | 0.984 | 0.984 | 0.984 | 0.996 |
| FaithDial | RoBERTa-base | Context + response | 3 | 0.958 ± 0.002 | 0.957 ± 0.002 | 0.958 ± 0.002 | 0.992 ± 0.000 |
| FaithDial | DeBERTa-v3-base | Context + response | 3 | 0.964 ± 0.002 | 0.963 ± 0.002 | 0.964 ± 0.002 | 0.994 ± 0.000 |
| FaithDial | RoBERTa-base | Response only | 3 | 0.937 ± 0.004 | 0.935 ± 0.005 | 0.937 ± 0.004 | 0.984 ± 0.001 |
| FaithDial | DeBERTa-v3-base | Response only | 3 | 0.935 ± 0.004 | 0.933 ± 0.005 | 0.935 ± 0.004 | 0.985 ± 0.001 |
| FaithDial | RoBERTa-base | Wrong context | 3 | 0.616 ± 0.022 | 0.720 ± 0.011 | 0.554 ± 0.034 | 0.942 ± 0.024 |
| FaithDial | DeBERTa-v3-base | Wrong context | 3 | 0.555 ± 0.006 | 0.689 ± 0.003 | 0.452 ± 0.011 | 0.921 ± 0.017 |
| FaithDial | RoBERTa-base | Trained on the other dataset | 3 | 0.486 ± 0.001 | 0.651 ± 0.003 | 0.338 ± 0.015 | 0.463 ± 0.048 |
| FaithDial | DeBERTa-v3-base | Trained on the other dataset | 3 | 0.508 ± 0.045 | 0.624 ± 0.027 | 0.455 ± 0.059 | 0.539 ± 0.019 |
| FaithDial | Baseline: length | Surface features only | 1 | 0.534 | 0.519 | 0.534 | 0.548 |
| FaithDial | Baseline: overlap | Surface features only | 1 | 0.561 | 0.561 | 0.561 | 0.584 |
