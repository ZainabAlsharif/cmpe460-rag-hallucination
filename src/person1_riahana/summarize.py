"""Collects the result files into Markdown and LaTeX tables and draws the report figures.

Usage (from the repository root):
    python -m src.person1_riahana.summarize
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .baselines import DATA_DIRS, words  # noqa: E402
from .datasets import load_examples  # noqa: E402

RESULTS = Path("results/person1_riahana")
FIGURES = Path("report/figures/person1")
TABLE = Path("report/tables/person1_results.tex")

DATASETS = {"halueval_qa": ("halueval", "HaluEval-QA"), "faithdial": ("faithdial", "FaithDial")}
MODELS = {"roberta": ("RoBERTa-base", "#2a78d6"), "deberta": ("DeBERTa-v3-base", "#eb6834")}
BASELINE_COLOR = "#8a8984"
LABEL_COLORS = {0: ("grounded", "#1baf7a"), 1: ("hallucinated", "#6250d6")}
INK, MUTED = "#0b0b0b", "#52514e"


METRICS = ("accuracy", "f1", "macro_f1", "auroc")


class ResultsSummary:
    """Repetitions of a run ("..._seed<number>") are grouped with it and reported as mean and standard deviation."""

    def __init__(self, results_dir: Path = RESULTS):
        runs = defaultdict(list)
        for path in sorted(results_dir.glob("p1_*.json")):
            name = re.sub(r"_seed\d+", "", path.stem)
            report = json.loads(path.read_text(encoding="utf-8"))["test"]
            if path.stem == name:       # the default seed goes first: its confusion matrix is plotted
                runs[name].insert(0, report)
            else:
                runs[name].append(report)
        self.results = {name: self._aggregate(reports) for name, reports in runs.items()}

    @staticmethod
    def _aggregate(reports):
        out = dict(reports[0], runs=len(reports))
        for metric in METRICS:
            values = [r[metric] for r in reports]
            out[metric] = float(np.mean(values))
            out[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        return out

    def get(self, name: str):
        return self.results.get(name)

    def settings(self, dataset: str):
        """(label, result name per model) for every way a model was tested on this dataset."""
        short = DATASETS[dataset][0]
        other = "faithdial" if dataset == "halueval_qa" else "halueval"
        return [
            ("Context + response", {m: f"p1_{m}_{short}" for m in MODELS}),
            ("Response only", {m: f"p1_{m}_{short}_answer_only" for m in MODELS}),
            ("Wrong context", {m: f"p1_{m}_{short}_wrongcontext" for m in MODELS}),
            ("Trained on the other dataset", {m: f"p1_{m}_{other}_on_{dataset}" for m in MODELS}),
        ]

    # ------------------------------------------------------------------ tables
    def rows(self):
        for dataset, (_, title) in DATASETS.items():
            for setting, names in self.settings(dataset):
                for model, name in names.items():
                    yield title, MODELS[model][0], setting, self.get(name)
            for baseline in ("length", "overlap"):
                yield title, f"Baseline: {baseline}", "Surface features only", \
                    self.get(f"p1_baseline_{baseline}_{dataset}")

    @staticmethod
    def _cells(report, plus_minus: str = " ± "):
        if report is None:
            return ["-"] * 4
        return [f"{report[k]:.3f}" + (f"{plus_minus}{report[k + '_std']:.3f}" if report["runs"] > 1 else "")
                for k in METRICS]

    def markdown(self) -> str:
        lines = ["# Person 1 results (test sets)", "",
                 "F1 is the F1 of the hallucinated class. Neural models: mean ± standard deviation over the "
                 "training seeds (column Runs). The baselines are deterministic.", "",
                 "| Dataset | Model | Setting | Runs | Acc | F1 | Macro-F1 | AUROC |", "|---|---|---|---|---|---|---|---|"]
        for dataset, model, setting, report in self.rows():
            runs = str(report["runs"]) if report else "-"
            lines.append("| " + " | ".join([dataset, model, setting, runs, *self._cells(report)]) + " |")
        return "\n".join(lines) + "\n"

    def latex(self) -> str:
        lines = [r"\begin{table}[h]", r"\centering", r"\small", r"\begin{tabular}{lllcccc}", r"\hline",
                 r"Dataset & Model & Setting & Acc & F1 & Macro-F1 & AUROC \\", r"\hline"]
        previous = None
        for dataset, model, setting, report in self.rows():
            if previous and dataset != previous:
                lines.append(r"\hline")
            previous = dataset
            lines.append(" & ".join([dataset, model, setting, *self._cells(report, r"$\pm$")]) + r" \\")
        lines += [r"\hline", r"\end{tabular}",
                  r"\caption{Test results of the response-level classifiers, their ablations and the surface "
                  r"baselines. F1 refers to the hallucinated class. Neural models are reported as mean $\pm$ "
                  r"standard deviation over the training seeds; the baselines are deterministic.}",
                  r"\label{tab:p1-results}", r"\end{table}"]
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ figures
    @staticmethod
    def _clean(ax):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c9c8c2")
        ax.tick_params(colors=MUTED, length=0)

    def figure_accuracy(self, path: Path):
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        for ax, (dataset, (_, title)) in zip(axes, DATASETS.items()):
            groups = self.settings(dataset)
            for i, (_, names) in enumerate(groups):
                for j, (model, name) in enumerate(names.items()):
                    report = self.get(name)
                    if report is None:
                        continue
                    x = i + (j - 0.5) * 0.38
                    error = 100 * report["accuracy_std"]
                    ax.bar(x, 100 * report["accuracy"], 0.36, color=MODELS[model][1],
                           yerr=error or None, error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3},
                           label=MODELS[model][0] if i == 0 else None)
                    ax.text(x, 100 * report["accuracy"] + error + 1.5, f"{100 * report['accuracy']:.1f}",
                            ha="center", fontsize=8, color=INK)
            for k, baseline in enumerate(("length", "overlap")):
                report = self.get(f"p1_baseline_{baseline}_{dataset}")
                if report:
                    x = len(groups) + k * 0.5
                    ax.bar(x, 100 * report["accuracy"], 0.4, color=BASELINE_COLOR,
                           label="Surface baseline" if k == 0 else None)
                    ax.text(x, 100 * report["accuracy"] + 1.5, f"{100 * report['accuracy']:.1f}",
                            ha="center", fontsize=8, color=INK)
            ax.axhline(50, color=MUTED, linewidth=0.8, linestyle=":")
            ax.set_ylim(0, 112)
            ax.set_ylabel("Test accuracy (%)", color=MUTED)
            ax.set_title(f"Tested on {title}", loc="left", color=INK)
            self._clean(ax)
        ticks = list(range(4)) + [4, 4.5]
        axes[1].set_xticks(ticks, ["Context +\nresponse", "Response\nonly", "Wrong\ncontext",
                                   "Trained on\nother dataset", "Length\nbaseline", "Overlap\nbaseline"], fontsize=8)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def figure_lengths(self, path: Path):
        """Response length by label: shows how much the two classes differ before any model is used."""
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for ax, (dataset, (_, title)) in zip(axes, DATASETS.items()):
            test = load_examples(dataset, DATA_DIRS[dataset])["test"]
            bins = np.arange(0, 42, 2)
            for label, (name, color) in LABEL_COLORS.items():
                lengths = np.clip([len(words(e.response)) for e in test if e.label == label], 0, 40)
                ax.hist(lengths, bins=bins, histtype="stepfilled", alpha=0.35, color=color)
                ax.hist(lengths, bins=bins, histtype="step", linewidth=2, color=color, label=name)
            ax.set_title(title, loc="left", color=INK)
            ax.set_xlabel("Words in the response (40 = 40 or more)", color=MUTED)
            self._clean(ax)
        axes[0].set_ylabel("Test examples", color=MUTED)
        axes[1].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def figure_confusion(self, path: Path):
        fig, axes = plt.subplots(2, 2, figsize=(6.4, 6))
        for row, (model, (model_title, _)) in zip(axes, MODELS.items()):
            for ax, (_, (short, title)) in zip(row, DATASETS.items()):
                report = self.get(f"p1_{model}_{short}")
                ax.set_title(f"{model_title}\n{title}", fontsize=9, color=INK)
                if report is None:
                    ax.axis("off")
                    continue
                matrix = np.array(report["confusion_matrix"])
                share = matrix / matrix.sum(axis=1, keepdims=True)
                ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, f"{matrix[i, j]}\n{share[i, j]:.1%}", ha="center", va="center",
                                fontsize=9, color="white" if share[i, j] > 0.55 else INK)
                names = ["grounded", "halluc."]
                ax.set_xticks([0, 1], names, fontsize=8)
                ax.set_yticks([0, 1], names, fontsize=8)
                ax.set_xlabel("Predicted", fontsize=8, color=MUTED)
                ax.set_ylabel("True", fontsize=8, color=MUTED)
                ax.tick_params(length=0)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def write(self):
        FIGURES.mkdir(parents=True, exist_ok=True)
        TABLE.parent.mkdir(parents=True, exist_ok=True)
        (RESULTS / "summary.md").write_text(self.markdown(), encoding="utf-8")
        TABLE.write_text(self.latex(), encoding="utf-8")
        self.figure_accuracy(FIGURES / "accuracy_by_setting.png")
        self.figure_lengths(FIGURES / "response_lengths.png")
        self.figure_confusion(FIGURES / "confusion_matrices.png")
        print(f"written: {RESULTS / 'summary.md'}, {TABLE}, figures in {FIGURES}")


if __name__ == "__main__":
    ResultsSummary().write()
