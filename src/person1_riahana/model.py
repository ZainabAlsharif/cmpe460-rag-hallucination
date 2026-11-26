"""Tokenisation and the encoder classifier (RoBERTa / DeBERTa-v3 with a two-class head)."""
from typing import List

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .datasets import Example


class PairEncoder:
    """Builds model inputs.

    "full" and "shuffled_context": the pair [context] [response]; only the context is truncated.
    "answer_only": the response alone, so the model cannot compare it with anything.
    """

    def __init__(self, model_name: str, max_length: int, input_mode: str = "full"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length, self.input_mode = max_length, input_mode

    def __call__(self, batch: List[Example]):
        responses = [e.response for e in batch]
        if self.input_mode == "answer_only":
            enc = self.tokenizer(responses, truncation=True, max_length=self.max_length,
                                 padding=True, return_tensors="pt")
        else:
            enc = self.tokenizer([e.context for e in batch], responses, truncation="only_first",
                                 max_length=self.max_length, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([e.label for e in batch])
        return enc

    def loader(self, examples: List[Example], batch_size: int, shuffle: bool = False) -> DataLoader:
        return DataLoader(examples, batch_size=batch_size, shuffle=shuffle, collate_fn=self)


def build_classifier(model_name_or_dir: str, gradient_checkpointing: bool = False):
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_dir, num_labels=2)
    model = model.float()          # some checkpoints are stored in half precision
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model
