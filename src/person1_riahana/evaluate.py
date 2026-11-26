"""Test an already trained classifier without further training.

    # does the model read the context? give every response a wrong context
    python -m src.person1_riahana.evaluate --config configs/person1_riahana/roberta_halueval.yaml --shuffle-context

    # does it transfer? model trained on HaluEval, tested on FaithDial
    python -m src.person1_riahana.evaluate --config configs/person1_riahana/roberta_halueval.yaml \
        --data-config configs/person1_riahana/roberta_faithdial.yaml
"""
import argparse
import json
from pathlib import Path

import torch

from .config import ExperimentConfig
from .datasets import load_examples, shuffle_contexts
from .evaluation import compute_report, predict_probabilities, save_results
from .model import PairEncoder, build_classifier


def trained_config(path: str, seed: int = None) -> ExperimentConfig:
    """The settings the run was really trained with (including command line overrides), if it was saved.

    `seed` selects a repetition that was trained with `train.py --seed`.
    """
    config = ExperimentConfig.from_yaml(path)
    if seed is not None:
        config.seed = seed
        config.run_name += f"_seed{seed}"
        config.output_dir += f"_seed{seed}"
    saved = Path(config.results_dir) / f"{config.run_name}.json"
    if saved.exists():
        return ExperimentConfig(**json.loads(saved.read_text(encoding="utf-8"))["config"])
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="config of the trained run")
    parser.add_argument("--data-config", help="config whose test set is used (default: the training dataset)")
    parser.add_argument("--shuffle-context", action="store_true")
    parser.add_argument("--seed", type=int, help="evaluate the repetition trained with this seed")
    args = parser.parse_args()

    run = trained_config(args.config, args.seed)
    data = ExperimentConfig.from_yaml(args.data_config) if args.data_config else run
    device = "cuda" if torch.cuda.is_available() else "cpu"

    examples = load_examples(data.dataset, data.data_dir, data.split_seed, data.history_turns)["test"]
    if args.shuffle_context:
        examples = shuffle_contexts(examples, data.split_seed)
    encoder = PairEncoder(run.output_dir, run.max_length, run.input_mode)
    model = build_classifier(run.output_dir).to(device)
    prob = predict_probabilities(model, encoder.loader(examples, run.batch_size * 2), device, run.amp_dtype)
    labels = [e.label for e in examples]
    report = compute_report(labels, prob)

    name = run.run_name
    if data.dataset != run.dataset:
        name += f"_on_{data.dataset}"
    if args.shuffle_context:
        name += "_wrongcontext"
    save_results(run.results_dir, name, {"trained_on": run.dataset, "tested_on": data.dataset,
                 "shuffled_context": args.shuffle_context, "test": report}, labels, prob)
    print(f"{name}: acc={report['accuracy']:.4f} f1={report['f1']:.4f} macro_f1={report['macro_f1']:.4f} "
          f"auroc={report['auroc']:.4f}")


if __name__ == "__main__":
    main()
