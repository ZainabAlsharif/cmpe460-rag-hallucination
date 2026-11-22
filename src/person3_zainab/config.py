"""Experiment configuration for Person 3 (Zainab): small LLM judges fine-tuned with LoRA.

Every training run is described by one YAML file in ``configs/person3_zainab/``.
The YAML keys map one-to-one onto the fields of :class:`ExperimentConfig`; any key
that is left out keeps the default value defined here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class ExperimentConfig:
    """All settings of one experiment (data, model, LoRA, optimisation, output)."""

    # --- identification -------------------------------------------------------
    run_name: str
    model_name: str
    dataset: str  # "ragbench" or "triviaplus"

    # --- data -----------------------------------------------------------------
    data_dir: Optional[str] = None  # RAGBench: folder that contains one sub-folder per subset
    data_file: Optional[str] = None  # TRIVIA+: path to triviaplus_dataset.parquet
    subsets: List[str] = field(default_factory=lambda: ["finqa", "techqa", "emanual"])
    max_finqa_samples: Optional[int] = 2000  # cap the (huge, 92%-supported) finqa training split
    label_column: str = "response_level_label_binary"  # TRIVIA+ label used for train/valid
    eval_label_column: Optional[str] = None  # TRIVIA+ label used for test (defaults to label_column)
    max_train_samples: Optional[int] = None  # debugging: use only N training rows
    max_val_samples: Optional[int] = 600  # validation rows used during training (speed)
    max_test_samples: Optional[int] = None  # debugging: use only N test rows

    # --- input encoding -------------------------------------------------------
    max_length: int = 1536  # total tokens per example (context is truncated first)
    max_question_tokens: int = 256
    max_response_tokens: int = 512

    # --- model / quantisation -------------------------------------------------
    load_in_4bit: bool = True
    gradient_checkpointing: bool = True

    # --- LoRA -----------------------------------------------------------------
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: list(DEFAULT_LORA_TARGETS))

    # --- optimisation ---------------------------------------------------------
    learning_rate: float = 2.0e-4
    batch_size: int = 1
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    epochs: float = 1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    use_class_weights: bool = True
    fp16: bool = True
    seed: int = 42
    logging_steps: int = 10

    # --- output ---------------------------------------------------------------
    output_dir: str = "outputs/run"  # checkpoints + LoRA adapter (git-ignored)
    results_dir: str = "results/person3_zainab"  # small JSON/CSV results (committed)

    def __post_init__(self) -> None:
        if self.dataset not in {"ragbench", "triviaplus"}:
            raise ValueError(f"dataset must be 'ragbench' or 'triviaplus', got {self.dataset!r}")
        if self.dataset == "ragbench" and not self.data_dir:
            raise ValueError("RAGBench configs need 'data_dir'")
        if self.dataset == "triviaplus" and not self.data_file:
            raise ValueError("TRIVIA+ configs need 'data_file'")
        if self.eval_label_column is None:
            self.eval_label_column = self.label_column

    # ------------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "ExperimentConfig":
        """Read a YAML file; ``overrides`` (e.g. from the command line) win over the file."""
        with open(path, "r", encoding="utf-8") as handle:
            raw: Dict[str, Any] = yaml.safe_load(handle) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {unknown}")
        return cls(**raw)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
