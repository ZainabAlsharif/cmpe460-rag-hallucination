"""Evaluate a judge on a test split without training it (again).

Two uses:

1. **Transfer** - a judge trained on RAGBench (GPT-4 labels) tested on TRIVIA+ (human labels)::

       python -m src.person3_zainab.evaluate --run-dir outputs/p3_qwen3_ragbench \\
           --data-config configs/person3_zainab/qwen3_triviaplus.yaml \\
           --name p3_qwen3_transfer_ragbench_to_triviaplus

   The model settings and the decision threshold come from the trained run
   (``<run-dir>/config.yaml`` and ``<run-dir>/threshold.json``); only the dataset settings
   come from ``--data-config``.

2. **Zero-shot baseline** - the base model with no fine-tuning at all (threshold 0.5)::

       python -m src.person3_zainab.evaluate --zero-shot \\
           --data-config configs/person3_zainab/qwen3_triviaplus.yaml --name p3_qwen3_zeroshot_triviaplus
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

from .config import ExperimentConfig  # noqa: E402
from .data import load_splits  # noqa: E402
from .evaluation import Predictor, compute_report  # noqa: E402
from .model import JudgeModelFactory  # noqa: E402
from .prompt import PromptEncoder  # noqa: E402
from .train import save_results  # noqa: E402


class JudgeEvaluation:
    """Scores a trained adapter (or the bare base model) on the test split of a dataset."""

    DATA_KEYS = ("dataset", "data_dir", "data_file", "subsets", "max_finqa_samples",
                 "label_column", "eval_label_column", "max_test_samples")

    def __init__(self, data_config: str, name: str, run_dir: Optional[str] = None,
                 max_test_samples: Optional[int] = None):
        data_cfg = ExperimentConfig.from_yaml(data_config, max_test_samples=max_test_samples)
        self.run_dir = Path(run_dir) if run_dir else None
        # Model / encoding settings: from the trained run, or from data_config for zero-shot.
        self.model_cfg = ExperimentConfig.from_yaml(self.run_dir / "config.yaml") if self.run_dir else data_cfg
        merged = self.model_cfg.to_dict()
        merged.update({k: getattr(data_cfg, k) for k in self.DATA_KEYS})
        merged.update(run_name=name, max_train_samples=1, max_val_samples=1)  # only the test split is used
        self.cfg = ExperimentConfig(**merged)
        self.name = name
        self.threshold = self._load_threshold()

    def _load_threshold(self) -> float:
        if self.run_dir and (self.run_dir / "threshold.json").exists():
            return float(json.loads((self.run_dir / "threshold.json").read_text())["threshold"])
        return 0.5

    def run(self) -> dict:
        started = time.time()
        splits = load_splits(self.cfg)
        adapter = self.run_dir / "adapter" if self.run_dir else None
        model, tokenizer, verbalizer = JudgeModelFactory.load_for_inference(self.model_cfg, adapter)
        encoder = PromptEncoder(tokenizer, self.cfg.max_length, self.cfg.max_question_tokens, self.cfg.max_response_tokens)
        test_ds, stats = encoder.to_dataset(splits.test)
        probs = Predictor(model, tokenizer, verbalizer, self.cfg.eval_batch_size).predict_proba(test_ds)
        trained_on = self.model_cfg.dataset if self.run_dir else "none (zero-shot)"
        results = {
            "run_name": self.name,
            "model_name": self.model_cfg.model_name,
            "dataset": trained_on,
            "evaluated_on": self.cfg.dataset,
            "train_label_column": (self.model_cfg.label_column if self.model_cfg.dataset == "triviaplus"
                                   else "adherence_score") if self.run_dir else None,
            "test_label_column": self.cfg.eval_label_column if self.cfg.dataset == "triviaplus" else "adherence_score",
            "trained_run": self.model_cfg.run_name if self.run_dir else None,
            "threshold": self.threshold,
            "data": {"test": splits.summary()["test"]},
            "encoding": {"test": stats.as_dict()},
            "test": compute_report(splits.test["label"], probs, splits.test["source"], self.threshold),
            "minutes": round((time.time() - started) / 60, 1),
        }
        save_results(results, splits.test, probs, Path(self.model_cfg.results_dir), self.name, self.threshold)
        t = results["test"]
        print(f"\nEVAL  acc={t['accuracy']:.4f}  f1={t['f1']:.4f}  macro_f1={t['macro_f1']:.4f}  auroc={t['auroc']}")
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a judge (trained or zero-shot) on a test split.")
    parser.add_argument("--data-config", required=True, help="config whose dataset / test split to use")
    parser.add_argument("--name", required=True, help="name for the result files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", help="output_dir of a finished train.py run")
    group.add_argument("--zero-shot", action="store_true", help="base model without any fine-tuning")
    parser.add_argument("--max-test-samples", type=int, help="debugging: use only N test rows")
    args = parser.parse_args()
    JudgeEvaluation(args.data_config, args.name, args.run_dir, args.max_test_samples).run()


if __name__ == "__main__":
    main()
