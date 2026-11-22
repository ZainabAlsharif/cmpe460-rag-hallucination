"""Shared evaluation metrics so every person reports numbers the same way."""
from typing import Dict, Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """Accuracy plus precision / recall / F1 for the *hallucinated* class (label 1).

    Use this for the shared results table (Acc and F1 columns).
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def confusion(y_true: Sequence[int], y_pred: Sequence[int]):
    """2x2 confusion matrix with rows = true label, columns = predicted label."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
