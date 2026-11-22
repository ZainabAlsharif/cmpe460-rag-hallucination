"""Hugging Face Trainer for the Yes/No LLM judge with a class-weighted loss.

* The loss is cross-entropy over the two verbaliser logits (" Yes" = supported,
  " No" = hallucinated), see :class:`model.Verbalizer`.
* RAGBench is imbalanced (~11% hallucinated). Each example's loss is multiplied by
  ``N / (2 * n_class)`` so both classes count equally. The weights are applied **per
  example** (``reduction="none"``): with batch size 1, ``CrossEntropyLoss(weight=...)``
  with mean reduction would divide the weight away again and have no effect.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from transformers import Trainer

from .model import Verbalizer


def balanced_class_weights(labels: Sequence[int], num_classes: int = 2) -> torch.Tensor:
    """``N / (num_classes * count_c)`` for every class c (1.0 everywhere for balanced data)."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    return torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float32)


class WeightedLossTrainer(Trainer):
    """Trainer whose model output is read through a :class:`Verbalizer`."""

    def __init__(self, *args, verbalizer: Verbalizer, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbalizer = verbalizer
        self.class_weights = class_weights
        # We return the mean loss of each micro-batch ourselves, so the Trainer must divide
        # by gradient_accumulation_steps as usual (instead of expecting a summed loss).
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        logits = self.verbalizer.class_logits(model, inputs["input_ids"], inputs["attention_mask"])
        per_example = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        if self.class_weights is not None:
            per_example = per_example * self.class_weights.to(logits.device)[labels]
        loss = per_example.mean()
        return (loss, {"logits": logits}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Return the 2-class logits (not the full vocabulary) so metrics can be computed."""
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return loss.detach(), None, None
        return loss.detach(), outputs["logits"].detach(), inputs["labels"].detach()
