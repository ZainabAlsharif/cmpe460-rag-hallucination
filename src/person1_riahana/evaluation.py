"""Inference and metrics (the shared binary_metrics plus AUROC and the confusion matrix)."""
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ..common import binary_metrics, confusion


def autocast(device: str, amp_dtype):
    """Mixed precision context for "fp16" or "bf16"; does nothing for None or on the CPU."""
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type=device, dtype=dtype, enabled=bool(amp_dtype) and device == "cuda")


@torch.no_grad()
def predict_probabilities(model, loader, device: str, amp_dtype=None) -> np.ndarray:
    """Returns P(hallucinated) for every example of the loader, in order."""
    model.eval()
    out = []
    for batch in loader:
        batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast(device, amp_dtype):
            logits = model(**batch).logits
        out.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def compute_report(y_true: List[int], prob: np.ndarray, threshold: float = 0.5) -> Dict:
    y_pred = (prob >= threshold).astype(int)
    report = {k: float(v) for k, v in binary_metrics(y_true, y_pred).items()}
    report["auroc"] = float(roc_auc_score(y_true, prob)) if len(set(y_true)) == 2 else None
    report["confusion_matrix"] = confusion(y_true, y_pred).tolist()
    report["n"] = len(y_true)
    return report


def save_results(results_dir: str, name: str, payload: Dict, y_true=None, prob=None) -> Path:
    folder = Path(results_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if prob is not None:
        with open(folder / f"{name}_predictions.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "label", "prob_hallucinated"])
            writer.writerows((i, int(t), f"{p:.6f}") for i, (t, p) in enumerate(zip(y_true, prob)))
    return path
