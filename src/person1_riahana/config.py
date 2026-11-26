"""Experiment configuration for the response-level encoder classifiers."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

INPUT_MODES = ("full", "answer_only", "shuffled_context")


@dataclass
class ExperimentConfig:
    """Settings of one run. Values come from a YAML file; command line flags can override them."""

    run_name: str
    model_name: str
    dataset: str                      # "halueval_qa" or "faithdial"
    data_dir: str
    output_dir: str
    max_length: int = 384
    learning_rate: float = 2.0e-5
    batch_size: int = 16
    grad_accumulation: int = 1
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = True
    gradient_checkpointing: bool = False
    bf16: bool = False                # bfloat16 mixed precision (stable for DeBERTa-v3, needs a recent GPU)
    eight_bit_optimizer: bool = False  # 8-bit AdamW from bitsandbytes, to fit DeBERTa-v3 on a 4 GB card
    seed: int = 42                    # training randomness: classifier head, batch order, dropout
    split_seed: int = 42              # data side: HaluEval split, subsampling, context shuffling
    input_mode: str = "full"          # what the model is allowed to read (see INPUT_MODES)
    history_turns: int = 3            # FaithDial: number of previous dialogue turns kept in the context
    max_train_examples: int = 0       # 0 = use everything (set a small number for smoke tests)
    max_eval_examples: int = 0
    results_dir: str = "results/person1_riahana"

    def __post_init__(self):
        if self.input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {INPUT_MODES}, got {self.input_mode!r}")
        if self.dataset not in ("halueval_qa", "faithdial"):
            raise ValueError(f"unknown dataset: {self.dataset!r}")

    @classmethod
    def from_yaml(cls, path, **overrides):
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        values.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
        return cls(**values)

    @property
    def amp_dtype(self):
        """"fp16", "bf16" or None (full precision)."""
        return "fp16" if self.fp16 else "bf16" if self.bf16 else None

    def to_dict(self):
        return asdict(self)
