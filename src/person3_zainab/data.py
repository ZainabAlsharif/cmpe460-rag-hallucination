"""Dataset loaders for Person 3: RAGBench (GPT-4 labels) and TRIVIA+ (human labels).

Both loaders return the same :class:`DataSplits` structure, a train / validation / test
triple of pandas DataFrames with exactly these columns:

=========  ===================================================================
context    the retrieved document(s) the answer should be grounded in
question   the user question
response   the LLM-generated answer that is checked
label      0 = supported / faithful, 1 = hallucinated / unfaithful
source     origin of the row (RAGBench subset or TRIVIA+ source dataset)
=========  ===================================================================

Run ``python data/person3_zainab/download.py`` first so the files exist locally.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .config import ExperimentConfig

COLUMNS = ["context", "question", "response", "label", "source"]
SUPPORTED, HALLUCINATED = 0, 1


@dataclass
class DataSplits:
    """Train / validation / test DataFrames in the common :data:`COLUMNS` format."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def items(self):
        return {"train": self.train, "validation": self.validation, "test": self.test}.items()

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Row count and hallucination rate per split (printed and stored with the results)."""
        return {
            name: {"rows": int(len(df)), "hallucinated_rate": round(float(df["label"].mean()), 4) if len(df) else 0.0}
            for name, df in self.items()
        }


def _sample(df: pd.DataFrame, n: Optional[int], seed: int) -> pd.DataFrame:
    """Randomly keep ``n`` rows, stratified by label so the class balance is preserved."""
    if n is None or len(df) <= n:
        return df.reset_index(drop=True)
    frac = n / len(df)
    parts = [group.sample(frac=frac, random_state=seed) for _, group in df.groupby("label")]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


class BaseLoader:
    """Common post-processing shared by both dataset loaders."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

    def load(self) -> DataSplits:
        splits = self._load_raw()
        cfg = self.cfg
        return DataSplits(
            train=_sample(self._clean(splits.train), cfg.max_train_samples, cfg.seed),
            validation=_sample(self._clean(splits.validation), cfg.max_val_samples, cfg.seed),
            test=_sample(self._clean(splits.test), cfg.max_test_samples, cfg.seed),
        )

    def _load_raw(self) -> DataSplits:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows without a label or without text, force types and keep only COLUMNS."""
        df = df.dropna(subset=["label"]).copy()
        df["label"] = df["label"].astype(int)
        for col in ("context", "question", "response"):
            df[col] = df[col].fillna("").astype(str).str.strip()
        df = df[(df["response"] != "") & (df["context"] != "")]
        return df[COLUMNS].reset_index(drop=True)


class RAGBenchLoader(BaseLoader):
    """Loads the RAGBench subsets saved by ``download.py`` (``datasets.save_to_disk`` format).

    Label: ``adherence_score`` (True = the response is grounded) → 0, False → 1.
    The list of retrieved ``documents`` is joined into one context string.
    """

    def _load_raw(self) -> DataSplits:
        from datasets import load_from_disk  # imported lazily: not needed for TRIVIA+

        frames: Dict[str, list] = {"train": [], "validation": [], "test": []}
        for subset in self.cfg.subsets:
            path = Path(self.cfg.data_dir) / subset
            if not path.exists():
                raise FileNotFoundError(f"{path} not found - run: python data/person3_zainab/download.py")
            dataset_dict = load_from_disk(str(path))
            for split in frames:
                df = self.convert(dataset_dict[split].to_pandas(), subset)
                if subset == "finqa" and split == "train":
                    df = _sample(df, self.cfg.max_finqa_samples, self.cfg.seed)
                frames[split].append(df)
        return DataSplits(*(pd.concat(frames[s], ignore_index=True) for s in ("train", "validation", "test")))

    @staticmethod
    def convert(raw: pd.DataFrame, subset: str) -> pd.DataFrame:
        """Map one raw RAGBench split onto the common columns."""
        raw = raw[raw["adherence_score"].notna()]
        return pd.DataFrame(
            {
                "context": raw["documents"].map(RAGBenchLoader.join_documents),
                "question": raw["question"],
                "response": raw["response"],
                "label": raw["adherence_score"].map(lambda grounded: SUPPORTED if bool(grounded) else HALLUCINATED),
                "source": subset,
            }
        )

    @staticmethod
    def join_documents(documents) -> str:
        """Join the retrieved passages, numbering them like a RAG prompt would."""
        if documents is None:
            return ""
        if isinstance(documents, str):
            return documents
        return "\n\n".join(f"[Document {i + 1}] {str(doc).strip()}" for i, doc in enumerate(documents))


class TriviaPlusLoader(BaseLoader):
    """Loads TRIVIA+ (``triviaplus_dataset.parquet``) and its official train / valid / test split.

    ``label_column`` is used for train and validation, ``eval_label_column`` for test. This is
    what makes the *LLM-judge labels vs. human labels* experiment possible: train on
    ``response_level_label_binary_ws_llm_aaj`` but always test on the human labels.
    """

    SPLIT_NAMES = {"train": "train", "validation": "valid", "test": "test"}

    def _load_raw(self) -> DataSplits:
        path = Path(self.cfg.data_file)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run: python data/person3_zainab/download.py")
        raw = pd.read_parquet(path)
        return DataSplits(
            train=self.convert(raw, "train", self.cfg.label_column),
            validation=self.convert(raw, "validation", self.cfg.label_column),
            test=self.convert(raw, "test", self.cfg.eval_label_column),
        )

    @classmethod
    def convert(cls, raw: pd.DataFrame, split: str, label_column: str) -> pd.DataFrame:
        if label_column not in raw.columns:
            raise KeyError(f"Label column {label_column!r} not in TRIVIA+ (columns: {list(raw.columns)})")
        part = raw[raw["split"] == cls.SPLIT_NAMES[split]]
        return pd.DataFrame(
            {
                "context": part["article"],
                "question": part["question"],
                "response": part["answer"],
                "label": pd.to_numeric(part[label_column], errors="coerce"),
                "source": part["source"],
            }
        )


def load_splits(cfg: ExperimentConfig) -> DataSplits:
    """Factory used by the training and evaluation scripts."""
    loader = {"ragbench": RAGBenchLoader, "triviaplus": TriviaPlusLoader}[cfg.dataset](cfg)
    return loader.load()
