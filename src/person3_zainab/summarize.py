"""Collect all Person 3 results into report-ready tables and figures.

Usage (from the repository root, after the runs have finished)::

    python -m src.person3_zainab.summarize

Reads ``results/person3_zainab/*.json`` and writes

* ``results/person3_zainab/summary.md``          markdown table (paste into results/results.md)
* ``report/tables/person3_results.tex``          LaTeX table for the report
* ``report/figures/person3/f1_by_run.png``       bar chart of test F1 / macro-F1
* ``report/figures/person3/confusion_matrices.png``
* ``report/figures/person3/f1_per_source.png``   F1 per domain / source dataset
* ``report/figures/person3/roc_curves.png``      ROC curves, one panel per test dataset
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATASET_NAMES = {"ragbench": "RAGBench", "triviaplus": "TRIVIA+"}

# Colour-blind-safe categorical palette (fixed order, never cycled) and text inks.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OVERFLOW = "#a3a29c"  # runs beyond the 8th share a neutral grey
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8,
})


def short_model(name: str) -> str:
    return name.split("/")[-1].replace("-Instruct", "")


class ResultsSummary:
    """Loads every result JSON and renders tables and plots."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.runs: List[Dict] = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]
        if not self.runs:
            raise SystemExit(f"No result files in {results_dir} - run train.py first.")

    def color(self, run_name: str) -> str:
        """Colour follows the run (entity), not its rank in a particular chart."""
        names = sorted(r["run_name"] for r in self.runs)
        idx = names.index(run_name)
        return PALETTE[idx] if idx < len(PALETTE) else OVERFLOW

    @staticmethod
    def style(ax) -> None:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    @staticmethod
    def describe(run: Dict) -> str:
        """Human-readable experiment description, e.g. 'RAGBench → TRIVIA+'."""
        tested = DATASET_NAMES.get(run.get("evaluated_on", run["dataset"]), run.get("evaluated_on"))
        if str(run["dataset"]).startswith("none"):
            return f"zero-shot on {tested}"
        trained = DATASET_NAMES.get(run["dataset"], run["dataset"])
        text = trained if trained == tested else f"{trained} → {tested}"
        if (run.get("train_label_column") or "").endswith("ws_llm_aaj"):
            text += " (LLM-judge labels)"
        return text

    def rows(self) -> List[Dict]:
        out = []
        for run in self.runs:
            t = run["test"]
            out.append({
                "run": run["run_name"], "model": short_model(run["model_name"]), "experiment": self.describe(run),
                "acc": t["accuracy"], "precision": t["precision"], "recall": t["recall"], "f1": t["f1"],
                "macro_f1": t["macro_f1"], "auroc": t["auroc"], "n": t["n"], "minutes": run.get("minutes"),
            })
        return out

    # ----------------------------------------------------------------- tables
    def markdown(self) -> str:
        lines = ["| Model | Train → Test | Acc | Precision | Recall | F1 | Macro-F1 | AUROC | n |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for r in self.rows():
            auroc = "–" if r["auroc"] is None else format(r["auroc"], ".3f")
            lines.append(f"| {r['model']} | {r['experiment']} | {r['acc']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} "
                         f"| {r['f1']:.3f} | {r['macro_f1']:.3f} | {auroc} | {r['n']} |")
        return "\n".join(lines)

    def latex(self) -> str:
        arrow = "$\\rightarrow$"
        end_row = " \\\\"
        lines = []
        for r in self.rows():
            experiment = r["experiment"].replace("→", arrow)
            auroc = "--" if r["auroc"] is None else format(r["auroc"], ".3f")
            lines.append(f"        {r['model']} & {experiment} & {r['acc']:.3f} & {r['f1']:.3f} & "
                         f"{r['macro_f1']:.3f} & {auroc}{end_row}")
        body = "\n".join(lines)
        return (
            "\\begin{table}[h]\n    \\centering\n    \\begin{tabular}{l l c c c c}\n"
            "        \\textbf{Model} & \\textbf{Train $\\rightarrow$ Test} & \\textbf{Acc} & \\textbf{F1} & "
            "\\textbf{Macro-F1} & \\textbf{AUROC} \\\\ \\hline\n"
            f"{body}\n    \\end{{tabular}}\n"
            "    \\caption[Person 3 results]{Test results of the LoRA-fine-tuned LLM judges. F1, precision and "
            "recall refer to the hallucinated class.}\n    \\label{tab:person3-results}\n\\end{table}\n"
        )

    # ---------------------------------------------------------------- figures
    def plot_f1(self, path: Path) -> None:
        rows = self.rows()
        labels = [f"{r['model']}\n{r['experiment']}" for r in rows]
        x = range(len(rows))
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(rows)), 4))
        ax.bar([i - 0.2 for i in x], [r["f1"] for r in rows], width=0.38, label="F1 (hallucinated class)",
               color=PALETTE[0], edgecolor="white", linewidth=2)
        ax.bar([i + 0.2 for i in x], [r["macro_f1"] for r in rows], width=0.38, label="Macro-F1",
               color=PALETTE[1], edgecolor="white", linewidth=2)
        ax.set_xticks(list(x), labels, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score on test set")
        ax.legend(frameon=False)
        self.style(ax)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def plot_confusions(self, path: Path) -> None:
        n = len(self.runs)
        cols = min(3, n)
        rows_n = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(3.2 * cols, 3.0 * rows_n), squeeze=False)
        for ax, run in zip(axes.flat, self.runs):
            cm = run["test"]["confusion_matrix"]
            ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=11)
            ax.set_xticks([0, 1], ["supported", "halluc."])
            ax.set_yticks([0, 1], ["supported", "halluc."])
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{short_model(run['model_name'])}\n{self.describe(run)}", fontsize=9)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def plot_per_source(self, path: Path) -> None:
        runs = [r for r in self.runs if r["test"].get("per_source")]
        if not runs:
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        for run in runs:
            per = run["test"]["per_source"]
            names = sorted(per)
            ax.plot(names, [per[s]["f1"] for s in names], marker="o", markersize=6, linewidth=2,
                    color=self.color(run["run_name"]),
                    label=f"{short_model(run['model_name'])} · {self.describe(run)}")
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1 (hallucinated class)")
        ax.set_xlabel("source / domain")
        ax.legend(fontsize=7, frameon=False)
        self.style(ax)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def plot_roc(self, path: Path) -> None:
        """ROC curve of every run, one panel per test dataset (reads the *_predictions.csv files)."""
        from sklearn.metrics import roc_curve

        panels: Dict[str, List[Dict]] = {}
        for run in self.runs:
            csv = self.results_dir / f"{run['run_name']}_predictions.csv"
            if csv.exists():
                panels.setdefault(run.get("evaluated_on", run["dataset"]), []).append(run)
        if not panels:
            return
        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6), squeeze=False)
        for ax, (tested, runs) in zip(axes[0], sorted(panels.items())):
            ax.plot([0, 1], [0, 1], linestyle="--", color=OVERFLOW, linewidth=1, label="chance (AUROC 0.5)")
            for run in runs:
                pred = pd.read_csv(self.results_dir / f"{run['run_name']}_predictions.csv")
                if pred["label"].nunique() < 2:
                    continue
                fpr, tpr, _ = roc_curve(pred["label"], pred["prob_hallucinated"])
                auroc = run["test"].get("auroc")
                ax.plot(fpr, tpr, linewidth=2, color=self.color(run["run_name"]),
                        label=f"{short_model(run['model_name'])} · {self.describe(run)}  (AUROC {auroc:.3f})")
            ax.set_title(f"Test set: {DATASET_NAMES.get(tested, tested)}", color=INK)
            ax.set_xlabel("false positive rate (supported answers flagged)")
            ax.set_ylabel("true positive rate (hallucinations caught)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.01)
            ax.legend(fontsize=7, frameon=False, loc="lower right")
            self.style(ax)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def write_all(self, table_dir: Path, figure_dir: Path) -> None:
        table_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        md = self.markdown()
        (self.results_dir / "summary.md").write_text(md + "\n", encoding="utf-8")
        (table_dir / "person3_results.tex").write_text(self.latex(), encoding="utf-8")
        self.plot_f1(figure_dir / "f1_by_run.png")
        self.plot_confusions(figure_dir / "confusion_matrices.png")
        self.plot_per_source(figure_dir / "f1_per_source.png")
        self.plot_roc(figure_dir / "roc_curves.png")
        print(md)
        print(f"\nWrote {self.results_dir / 'summary.md'}, {table_dir / 'person3_results.tex'} and figures in {figure_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise Person 3 results into tables and figures.")
    parser.add_argument("--results-dir", default="results/person3_zainab")
    parser.add_argument("--table-dir", default="report/tables")
    parser.add_argument("--figure-dir", default="report/figures/person3")
    args = parser.parse_args()
    ResultsSummary(Path(args.results_dir)).write_all(Path(args.table_dir), Path(args.figure_dir))


if __name__ == "__main__":
    main()
