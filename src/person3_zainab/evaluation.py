"""Prediction and metrics for the LLM judge.

* :class:`Predictor` runs a model over an encoded dataset and returns P(hallucinated).
* :func:`best_threshold` picks the decision threshold on the **validation** set.
* :func:`compute_report` turns labels + probabilities into the numbers used in the report:
  accuracy, precision / recall / F1 of the hallucinated class, macro-F1, AUROC, the
  confusion matrix and the same scores per source (RAGBench subset / TRIVIA+ source).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.common.metrics import binary_metrics, confusion


def compute_report(labels: Sequence[int], probs: Sequence[float], sources: Sequence[str] | None = None,
                   threshold: float = 0.5) -> Dict:
    """All evaluation numbers for one set of predictions."""
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)
    report = {k: round(float(v), 4) for k, v in binary_metrics(labels, preds).items()}
    report["auroc"] = round(float(roc_auc_score(labels, probs)), 4) if len(set(labels)) == 2 else None
    report["threshold"] = round(float(threshold), 4)
    report["n"] = int(len(labels))
    report["hallucinated_rate"] = round(float(labels.mean()), 4) if len(labels) else 0.0
    report["predicted_hallucinated_rate"] = round(float(preds.mean()), 4) if len(preds) else 0.0
    report["confusion_matrix"] = confusion(labels, preds).tolist()  # rows = true, cols = predicted
    if sources is not None:
        frame = pd.DataFrame({"label": labels, "prob": probs, "source": list(sources)})
        report["per_source"] = {
            str(name): compute_report(group["label"], group["prob"], None, threshold)
            for name, group in frame.groupby("source")
        }
    return report


def best_threshold(labels: Sequence[int], probs: Sequence[float]) -> float:
    """Threshold on P(hallucinated) that maximises macro-F1 on the given (validation) data.

    Macro-F1 is used instead of the F1 of the hallucinated class because the latter is
    maximised trivially by calling *everything* hallucinated.
    """
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    candidates = np.unique(np.concatenate([np.quantile(probs, np.linspace(0.02, 0.98, 49)), [0.5]]))
    scores = [f1_score(labels, (probs >= t).astype(int), average="macro", zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def trainer_metrics(eval_pred) -> Dict[str, float]:
    """``compute_metrics`` callback for the Hugging Face Trainer (validation set, threshold 0.5)."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    probs = softmax(np.asarray(logits, dtype=np.float64))[:, 1]
    report = compute_report(labels, probs)
    return {k: v for k, v in report.items() if isinstance(v, (int, float)) and v is not None}


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class Predictor:
    """Batched inference, longest examples first (out-of-memory errors show up immediately)."""

    def __init__(self, model, tokenizer, verbalizer, batch_size: int = 2):
        self.model = model
        self.tokenizer = tokenizer
        self.verbalizer = verbalizer
        self.batch_size = batch_size

    def predict_proba(self, dataset) -> np.ndarray:
        import torch
        from tqdm.auto import tqdm

        input_ids: List[List[int]] = dataset["input_ids"]
        order = np.argsort([len(ids) for ids in input_ids])[::-1]
        probs = np.zeros(len(input_ids), dtype=np.float64)
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            for start in tqdm(range(0, len(order), self.batch_size), desc="predict"):
                idx = order[start:start + self.batch_size]
                batch = self.tokenizer.pad({"input_ids": [input_ids[i] for i in idx]}, return_tensors="pt")
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = self.verbalizer.class_logits(self.model, batch["input_ids"], batch["attention_mask"])
                probs[idx] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        return probs
