"""Looks behind the headline scores, using the saved test predictions of all seeds (no training, no GPU).

1. Are the differences real?  Paired bootstrap over groups of related examples, and McNemar's test per seed.
2. Does the model react to the context?  Grounded responses flagged as hallucinated, right vs wrong context.
3. Is length the only shortcut?  Scores on HaluEval responses of similar length.
4. Which examples go wrong?  Error rates by kind of example, and a list of examples most runs get wrong.
5. Do training and test data overlap?  Counts of shared text, and FaithDial scores on unseen knowledge.

Usage (from the repository root):
    python -m src.person1_riahana.analysis
"""
import csv
import re
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from .baselines import DATA_DIRS, words
from .datasets import load_examples

RESULTS = Path("results/person1_riahana")
DATASETS = {"halueval_qa": ("halueval", "HaluEval-QA"), "faithdial": ("faithdial", "FaithDial")}
MODELS = {"roberta": "RoBERTa-base", "deberta": "DeBERTa-v3-base"}


class Analysis:
    def __init__(self, results_dir: Path = RESULTS, bootstrap_samples: int = 10000, seed: int = 42):
        self.results_dir, self.samples, self.seed = results_dir, bootstrap_samples, seed
        self.test = {d: load_examples(d, DATA_DIRS[d], seed)["test"] for d in DATASETS}
        self.labels = {d: np.array([e.label for e in v]) for d, v in self.test.items()}

    # ------------------------------------------------------------------ loading
    def probabilities(self, dataset: str, model: str, setting: str = "") -> np.ndarray:
        """P(hallucinated), shape (seeds, examples). setting: "", "_answer_only" or "_wrongcontext"."""
        base = f"p1_{model}_{DATASETS[dataset][0]}"
        mode, context = ("_answer_only", "") if setting == "_answer_only" else ("", setting)
        pattern = re.compile(rf"{base}{mode}(_seed\d+)?{context}_predictions\.csv")
        paths = sorted(p for p in self.results_dir.glob(f"{base}*_predictions.csv") if pattern.fullmatch(p.name))
        if not paths:
            raise FileNotFoundError(f"no predictions for {base}{setting}")
        runs = []
        for path in paths:
            with open(path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not np.array_equal([int(r["label"]) for r in rows], self.labels[dataset]):
                raise ValueError(f"{path.name} does not match the test set")
            runs.append([float(r["prob_hallucinated"]) for r in rows])
        return np.array(runs)

    def correct(self, dataset: str, model: str, setting: str = "") -> np.ndarray:
        return (self.probabilities(dataset, model, setting) >= 0.5) == self.labels[dataset]

    # ------------------------------------------------------------------ 1. differences
    def groups(self, dataset: str) -> np.ndarray:
        """Related test examples: the same knowledge passage (HaluEval) or the same dialogue turn (FaithDial)."""
        attribute = "knowledge" if dataset == "halueval_qa" else "context"
        ids = {}
        return np.array([ids.setdefault(getattr(e, attribute), len(ids)) for e in self.test[dataset]])

    def bootstrap_interval(self, difference: np.ndarray, groups: np.ndarray):
        """95% interval of the mean per-example difference, resampling whole groups."""
        totals = np.bincount(groups, weights=difference)
        sizes = np.bincount(groups)
        rng = np.random.default_rng(self.seed)
        means = np.empty(self.samples)
        for i in range(self.samples):
            pick = rng.integers(0, len(sizes), len(sizes))
            means[i] = totals[pick].sum() / sizes[pick].sum()
        return np.percentile(means, [2.5, 97.5])

    @staticmethod
    def mcnemar(a: np.ndarray, b: np.ndarray) -> float:
        """Exact two-sided McNemar test on two vectors of correct / wrong."""
        only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
        if only_a + only_b == 0:
            return 1.0
        return binomtest(only_a, only_a + only_b, 0.5).pvalue

    def comparisons(self):
        for dataset in DATASETS:
            yield dataset, "DeBERTa vs RoBERTa", ("deberta", ""), ("roberta", "")
            for model, name in MODELS.items():
                yield dataset, f"{name}: full input vs response only", (model, ""), (model, "_answer_only")

    def differences_section(self):
        lines = ["## 1. Are the differences real?", "",
                 "Accuracy is averaged over the seeds. The interval comes from a paired bootstrap that resamples "
                 "groups of related examples (same passage or same dialogue turn); it describes the uncertainty "
                 "from the choice of test examples, not from retraining. McNemar p-values are given for each seed "
                 "pair separately; that test treats the examples as independent. No correction for multiple "
                 "comparisons is applied.", "",
                 "| Dataset | Comparison (A vs B) | Acc A | Acc B | A - B (points) | 95% interval | McNemar p per seed |",
                 "|---|---|---|---|---|---|---|"]
        for dataset, title, first, second in self.comparisons():
            a, b = self.correct(dataset, *first), self.correct(dataset, *second)
            difference = a.mean(axis=0) - b.mean(axis=0)
            low, high = self.bootstrap_interval(difference, self.groups(dataset))
            p_values = ", ".join(f"{self.mcnemar(x, y):.2g}" for x, y in zip(a, b))
            lines.append(f"| {DATASETS[dataset][1]} | {title} | {100 * a.mean():.2f} | {100 * b.mean():.2f} | "
                         f"{100 * difference.mean():+.2f} | [{100 * low:+.2f}, {100 * high:+.2f}] | {p_values} |")
        return lines

    # ------------------------------------------------------------------ 2. context
    def context_section(self):
        lines = ["## 2. Does the model react to the context?", "",
                 "Share of grounded responses (label 0) that are called hallucinated. Mean ± standard deviation "
                 "over the seeds.", "",
                 "| Dataset | Model | Right context | Wrong context |", "|---|---|---|---|"]
        for dataset, (_, title) in DATASETS.items():
            grounded = self.labels[dataset] == 0
            for model, name in MODELS.items():
                cells = [self._mean_std((self.probabilities(dataset, model, s)[:, grounded] >= 0.5).mean(axis=1))
                         for s in ("", "_wrongcontext")]
                lines.append(f"| {title} | {name} | {cells[0]} | {cells[1]} |")
        return lines

    @staticmethod
    def _mean_std(values) -> str:
        values = 100 * np.asarray(values)
        return f"{values.mean():.1f} ± {values.std(ddof=1) if len(values) > 1 else 0.0:.1f}%"

    # ------------------------------------------------------------------ 3. length
    def length_section(self, low: int = 4, high: int = 6):
        dataset = "halueval_qa"
        labels = self.labels[dataset]
        keep = np.array([low <= len(words(e.response)) <= high for e in self.test[dataset]])
        lines = ["## 3. Is length the only shortcut?", "",
                 f"HaluEval-QA test responses with {low} to {high} words: {keep.sum()} examples, "
                 f"{int(labels[keep].sum())} of them hallucinated. Balanced accuracy is the mean of the two "
                 "per-class accuracies.", "",
                 "| Model | Input | Accuracy | Balanced accuracy |", "|---|---|---|---|"]
        rows = [(name, text, self.probabilities(dataset, model, setting))
                for model, name in MODELS.items()
                for setting, text in (("", "context + response"), ("_answer_only", "response only"))]
        for baseline in ("length", "overlap"):
            path = self.results_dir / f"p1_baseline_{baseline}_{dataset}_predictions.csv"
            with open(path, encoding="utf-8") as f:
                prob = np.array([[float(r["prob_hallucinated"]) for r in csv.DictReader(f)]])
            rows.append((f"Baseline: {baseline}", "surface features", prob))
        for name, text, prob in rows:
            correct = (prob[:, keep] >= 0.5) == labels[keep]
            balanced = np.mean([correct[:, labels[keep] == c].mean(axis=1) for c in (0, 1)], axis=0)
            lines.append(f"| {name} | {text} | {self._mean_std(correct.mean(axis=1))} | {self._mean_std(balanced)} |")
        return lines

    # ------------------------------------------------------------------ 4. errors
    def slices(self, dataset: str):
        examples, labels = self.test[dataset], self.labels[dataset]
        length = np.array([len(words(e.response)) for e in examples])
        copied = np.array([" ".join(words(e.response)) in " ".join(words(e.knowledge)) for e in examples])
        yield "All", np.ones(len(examples), bool)
        yield "Grounded (error = false alarm)", labels == 0
        yield "Hallucinated (error = miss)", labels == 1
        for text, mask in (("1-3 words", length <= 3), ("4-10 words", (length >= 4) & (length <= 10)),
                           ("11-20 words", (length >= 11) & (length <= 20)), ("21+ words", length >= 21)):
            yield text, mask
        yield "Grounded, copied word for word from the knowledge", (labels == 0) & copied
        yield "Grounded, not copied word for word", (labels == 0) & ~copied
        yield "Hallucinated, copied word for word from the knowledge", (labels == 1) & copied
        yield "Hallucinated, not copied word for word", (labels == 1) & ~copied

    def errors_section(self):
        lines = ["## 4. Which examples go wrong?", "",
                 "Error rate of the full-input models, averaged over the seeds."]
        for dataset, (_, title) in DATASETS.items():
            correct = {m: self.correct(dataset, m) for m in MODELS}
            lines += ["", f"### {title}", "", "| Slice | Examples | " + " | ".join(MODELS.values()) + " |",
                      "|---|---|" + "---|" * len(MODELS)]
            for text, mask in self.slices(dataset):
                if mask.sum() == 0:
                    continue
                cells = [f"{100 * (1 - correct[m][:, mask].mean()):.2f}%" for m in MODELS]
                lines.append(f"| {text} | {mask.sum()} | " + " | ".join(cells) + " |")
            wrong = sum((~c).sum(axis=0) for c in correct.values())
            runs = sum(len(c) for c in correct.values())
            hard = np.flatnonzero(wrong >= runs - 1)
            lines += ["", f"{(wrong == 0).sum()} examples are right in all {runs} runs; {len(hard)} are wrong in at "
                          f"least {runs - 1} runs (listed in `hard_examples_{dataset}.csv`)."]
            self._write_hard(dataset, hard, wrong, runs)
        return lines

    def _write_hard(self, dataset: str, indices, wrong, runs: int):
        with open(self.results_dir / f"hard_examples_{dataset}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "label", "wrong_runs", "total_runs", "context", "response"])
            for i in indices:
                e = self.test[dataset][i]
                writer.writerow([i, e.label, wrong[i], runs, e.context, e.response])

    # ------------------------------------------------------------------ 5. overlap
    def overlap_section(self):
        lines = ["## 5. Do training and test data overlap?", "",
                 "Number of test examples whose text also occurs in the training split.", "",
                 "| Dataset | Test examples | Same context and response | Same context | Same knowledge text | "
                 "Same response string |", "|---|---|---|---|---|---|"]
        known = {}
        for dataset, (_, title) in DATASETS.items():
            train = load_examples(dataset, DATA_DIRS[dataset], self.seed)["train"]
            known[dataset] = {e.knowledge for e in train}
            cells = []
            for key in (lambda e: (e.context, e.response), lambda e: e.context, lambda e: e.knowledge,
                        lambda e: e.response):
                seen = {key(e) for e in train}
                cells.append(str(sum(key(e) in seen for e in self.test[dataset])))
            lines.append(f"| {title} | {len(self.test[dataset])} | " + " | ".join(cells) + " |")

        dataset = "faithdial"
        seen = np.array([e.knowledge in known[dataset] for e in self.test[dataset]])
        lines += ["", "The official FaithDial splits share knowledge sentences, so its test accuracy is split by "
                      "whether the sentence occurs in the training data (mean ± standard deviation over the seeds).",
                  "", f"| Model | Input | Seen knowledge ({seen.sum()}) | Unseen knowledge ({(~seen).sum()}) |",
                  "|---|---|---|---|"]
        for model, name in MODELS.items():
            for setting, text in (("", "context + response"), ("_answer_only", "response only")):
                correct = self.correct(dataset, model, setting)
                cells = [self._mean_std(correct[:, mask].mean(axis=1)) for mask in (seen, ~seen)]
                lines.append(f"| {name} | {text} | {cells[0]} | {cells[1]} |")
        return lines

    def write(self) -> Path:
        sections = (self.differences_section(), self.context_section(), self.length_section(), self.errors_section(),
                    self.overlap_section())
        lines = ["# Person 1: analysis of the test predictions"]
        for section in sections:
            lines += ["", *section]
        path = self.results_dir / "analysis.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


if __name__ == "__main__":
    print(f"written: {Analysis().write()}")
