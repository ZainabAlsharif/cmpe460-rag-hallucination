"""Train one LLM judge (Person 3) and evaluate it on the test split.

Usage (from the repository root, on a GPU machine such as a Kaggle T4)::

    python -m src.person3_zainab.train --config configs/person3_zainab/qwen3_triviaplus.yaml

Outputs
-------
* ``<output_dir>/adapter/``     trained LoRA adapter (git-ignored)
* ``<output_dir>/threshold.json`` decision threshold tuned on the validation split
* ``<output_dir>/config.yaml``  the exact configuration that was used
* ``<results_dir>/<run_name>.json``             all metrics (committed to git)
* ``<results_dir>/<run_name>_predictions.csv``  per-example test predictions (no text)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # 4-bit models must sit on a single GPU
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
from transformers import DataCollatorWithPadding, TrainingArguments  # noqa: E402

from src.common.seed import set_seed  # noqa: E402

from .config import ExperimentConfig  # noqa: E402
from .data import DataSplits, load_splits  # noqa: E402
from .evaluation import Predictor, best_threshold, compute_report, trainer_metrics  # noqa: E402
from .model import JudgeModelFactory  # noqa: E402
from .prompt import PromptEncoder  # noqa: E402
from .trainer import WeightedLossTrainer, balanced_class_weights  # noqa: E402


class Experiment:
    """One complete run: data → encoding → LoRA fine-tuning → threshold on validation → test → results."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.adapter_dir = self.output_dir / "adapter"
        self.results_dir = Path(cfg.results_dir)

    # ------------------------------------------------------------------ steps
    def load_data(self) -> DataSplits:
        splits = load_splits(self.cfg)
        print("Data:", json.dumps(splits.summary(), indent=2))
        return splits

    def build_training_args(self) -> TrainingArguments:
        cfg = self.cfg
        return TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.eval_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit" if cfg.load_in_4bit else "adamw_torch",
            fp16=cfg.fp16,
            logging_steps=cfg.logging_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",  # F1 of one class rewards "always hallucinated"
            greater_is_better=True,
            group_by_length=True,  # batches of similar length → less padding
            label_names=["labels"],  # PEFT hides the label argument; tell the Trainer explicitly
            report_to="none",
            seed=cfg.seed,
        )

    def run(self) -> dict:
        cfg = self.cfg
        set_seed(cfg.seed)
        started = time.time()
        splits = self.load_data()

        model, tokenizer, verbalizer = JudgeModelFactory.build_for_training(cfg)
        params = JudgeModelFactory.count_parameters(model)
        encoder = PromptEncoder(tokenizer, cfg.max_length, cfg.max_question_tokens, cfg.max_response_tokens)
        train_ds, train_stats = encoder.to_dataset(splits.train)
        val_ds, val_stats = encoder.to_dataset(splits.validation)
        test_ds, test_stats = encoder.to_dataset(splits.test)
        print("Encoding (train):", train_stats.as_dict())

        weights = balanced_class_weights(splits.train["label"]) if cfg.use_class_weights else None
        print("Class weights:", None if weights is None else weights.tolist())

        trainer = WeightedLossTrainer(
            model=model,
            args=self.build_training_args(),
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
            compute_metrics=trainer_metrics,
            verbalizer=verbalizer,
            class_weights=weights,
        )
        train_output = trainer.train()

        # Save the best adapter and the config used.
        trainer.save_model(str(self.adapter_dir))
        tokenizer.save_pretrained(str(self.adapter_dir))
        cfg.save(self.output_dir / "config.yaml")

        # Decision threshold: tuned on VALIDATION only, then applied unchanged to test.
        predictor = Predictor(trainer.model, tokenizer, verbalizer, cfg.eval_batch_size)
        val_probs = predictor.predict_proba(val_ds)
        threshold = best_threshold(splits.validation["label"], val_probs)
        (self.output_dir / "threshold.json").write_text(json.dumps({"threshold": threshold}))
        test_probs = predictor.predict_proba(test_ds)

        results = {
            "run_name": cfg.run_name,
            "model_name": cfg.model_name,
            "dataset": cfg.dataset,
            "evaluated_on": cfg.dataset,
            "train_label_column": cfg.label_column if cfg.dataset == "triviaplus" else "adherence_score",
            "test_label_column": cfg.eval_label_column if cfg.dataset == "triviaplus" else "adherence_score",
            "data": splits.summary(),
            "encoding": {"train": train_stats.as_dict(), "validation": val_stats.as_dict(), "test": test_stats.as_dict()},
            "parameters": params,
            "class_weights": None if weights is None else [round(w, 4) for w in weights.tolist()],
            "train_loss": round(float(train_output.training_loss), 4),
            "epochs_log": self._eval_history(trainer),
            "threshold": round(threshold, 4),
            "validation": compute_report(splits.validation["label"], val_probs, None, threshold),
            "test": compute_report(splits.test["label"], test_probs, splits.test["source"], threshold),
            "test_at_0.5": compute_report(splits.test["label"], test_probs, None, 0.5),
            "minutes": round((time.time() - started) / 60, 1),
            "config": cfg.to_dict(),
        }
        self.save_results(results, splits.test, test_probs, threshold)
        return results

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _eval_history(trainer) -> list:
        """Validation scores after every epoch (the best one by macro-F1 is the model kept)."""
        keep = ("epoch", "eval_loss", "eval_accuracy", "eval_f1", "eval_macro_f1", "eval_auroc",
                "eval_predicted_hallucinated_rate")
        return [{k.replace("eval_", ""): log[k] for k in keep if k in log}
                for log in trainer.state.log_history if "eval_macro_f1" in log]

    def save_results(self, results: dict, test_df: pd.DataFrame, probs, threshold: float) -> None:
        save_results(results, test_df, probs, self.results_dir, results["run_name"], threshold)
        test = results["test"]
        print(f"\nTEST  acc={test['accuracy']:.4f}  f1={test['f1']:.4f}  macro_f1={test['macro_f1']:.4f}  "
              f"auroc={test['auroc']}  threshold={results['threshold']}  ({results['minutes']} min)")


def save_results(results: dict, test_df: pd.DataFrame, probs, results_dir: Path, name: str,
                 threshold: float = 0.5) -> None:
    """Write ``<name>.json`` (metrics) and ``<name>_predictions.csv`` (no text, only labels/scores)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{name}.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    pd.DataFrame(
        {
            "row": range(len(test_df)),
            "source": test_df["source"].values,
            "label": test_df["label"].values,
            "prob_hallucinated": [round(float(p), 5) for p in probs],
            "pred": [int(p >= threshold) for p in probs],
        }
    ).to_csv(results_dir / f"{name}_predictions.csv", index=False)
    print(f"Saved results → {results_dir / (name + '.json')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a small LLM judge with LoRA (Person 3).")
    parser.add_argument("--config", required=True, help="YAML file in configs/person3_zainab/")
    parser.add_argument("--model-name", help="override model_name (e.g. Qwen/Qwen3-0.6B for a quick test)")
    parser.add_argument("--max-train-samples", type=int, help="override: use only N training rows")
    parser.add_argument("--max-test-samples", type=int, help="override: use only N test rows")
    parser.add_argument("--epochs", type=float, help="override number of epochs")
    parser.add_argument("--lora-r", type=int, help="override LoRA rank (rank ablation)")
    parser.add_argument("--run-name", help="override run name (and output/result file names)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "model_name": args.model_name,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "epochs": args.epochs,
        "lora_r": args.lora_r,
    }
    if args.run_name:
        overrides.update(run_name=args.run_name, output_dir=f"outputs/{args.run_name}")
    cfg = ExperimentConfig.from_yaml(args.config, **overrides)
    if args.lora_r and not args.run_name:  # keep rank-ablation runs from overwriting the main run
        cfg.run_name = f"{cfg.run_name}_r{args.lora_r}"
        cfg.output_dir = f"{cfg.output_dir}_r{args.lora_r}"
        cfg.lora_alpha = 2 * args.lora_r
    Experiment(cfg).run()


if __name__ == "__main__":
    main()
