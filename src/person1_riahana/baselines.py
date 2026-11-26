"""Shortcut baselines: logistic regression on a few surface features, no neural network.

"length": only the number of words and characters of the response.
"overlap": only how much of the response is copied from the reference text.
If a baseline like this scores high, the dataset can be solved without understanding the context.

Usage (from the repository root):
    python -m src.person1_riahana.baselines
"""
import argparse
import re
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .datasets import Example, load_examples
from .evaluation import compute_report, save_results

DATA_DIRS = {"halueval_qa": "data/person1_riahana/halueval_qa", "faithdial": "data/person1_riahana/faithdial"}


def words(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def length_features(e: Example) -> List[float]:
    return [len(words(e.response)), len(e.response)]


def overlap_features(e: Example) -> List[float]:
    response, knowledge = words(e.response), set(words(e.knowledge))
    covered = sum(w in knowledge for w in response) / max(len(response), 1)
    verbatim = float(" ".join(response) in " ".join(words(e.knowledge)))
    return [covered, verbatim]


BASELINES = {"length": length_features, "overlap": overlap_features}


def run_baseline(dataset: str, name: str, results_dir: str, seed: int = 42):
    data = load_examples(dataset, DATA_DIRS[dataset], seed)
    featurise = BASELINES[name]
    matrix = {s: np.array([featurise(e) for e in v]) for s, v in data.items()}
    labels = {s: [e.label for e in v] for s, v in data.items()}
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(matrix["train"], labels["train"])
    prob = model.predict_proba(matrix["test"])[:, 1]
    report = compute_report(labels["test"], prob)
    save_results(results_dir, f"p1_baseline_{name}_{dataset}",
                 {"baseline": name, "dataset": dataset, "test": report}, labels["test"], prob)
    print(f"{name:8s} {dataset:12s} acc={report['accuracy']:.4f} f1={report['f1']:.4f} auroc={report['auroc']:.4f}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/person1_riahana")
    args = parser.parse_args()
    for dataset in DATA_DIRS:
        for name in BASELINES:
            run_baseline(dataset, name, args.results_dir)


if __name__ == "__main__":
    main()
