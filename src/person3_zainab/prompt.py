"""Turns (context, question, response) rows into token ids for the LLM judge.

The judge sees one prompt per example::

    You are checking an answer produced by a retrieval-augmented generation (RAG) system.

    Context:
    <retrieved documents, truncated to fit>

    Question: <question>

    Answer: <LLM response>

    Is every claim in the answer supported by the context? Reply Yes or No.
    Verdict (Yes or No):

The model's next-token scores for " Yes" / " No" are the class scores
(0 = supported, 1 = hallucinated), see ``model.Verbalizer``. Only the *context* is shortened when the prompt is too
long; the question and the answer are kept (each up to its own cap), because a judge
cannot decide anything if the answer itself is cut off.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

HEADER = "You are checking an answer produced by a retrieval-augmented generation (RAG) system.\n\nContext:\n"
QUESTION = "\n\nQuestion: "
ANSWER = "\n\nAnswer: "
INSTRUCTION = "\n\nIs every claim in the answer supported by the context? Reply Yes or No.\nVerdict (Yes or No):"


@dataclass
class EncodingStats:
    """How much text had to be cut — reported in the results (useful for the Discussion)."""

    examples: int = 0
    truncated_contexts: int = 0
    mean_length: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        rate = self.truncated_contexts / self.examples if self.examples else 0.0
        return {"examples": self.examples, "truncated_context_rate": round(rate, 4), "mean_tokens": round(self.mean_length, 1)}


class PromptEncoder:
    """Builds the judge prompt directly as token ids, truncating only the context."""

    def __init__(self, tokenizer, max_length: int, max_question_tokens: int = 256, max_response_tokens: int = 512):
        self.tok = tokenizer
        self.max_length = max_length
        self.max_question_tokens = max_question_tokens
        self.max_response_tokens = max_response_tokens
        # Special tokens the model expects at the start (Llama: <|begin_of_text|>, Qwen: none).
        self.prefix_ids: List[int] = self._ids("", add_special_tokens=True)
        self.header_ids = self._ids(HEADER)
        self.question_ids = self._ids(QUESTION)
        self.answer_ids = self._ids(ANSWER)
        self.instruction_ids = self._ids(INSTRUCTION)
        fixed = len(self.prefix_ids) + len(self.header_ids) + len(self.question_ids)
        fixed += len(self.answer_ids) + len(self.instruction_ids)
        if fixed + max_question_tokens + max_response_tokens >= max_length:
            raise ValueError("max_length is too small for the question/answer caps - increase max_length")

    def _ids(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self.tok(text, add_special_tokens=add_special_tokens)["input_ids"]

    def encode(self, context: str, question: str, response: str) -> tuple[List[int], bool]:
        """Return (input_ids, context_was_truncated) for one example."""
        question_ids = self._ids(question)[: self.max_question_tokens]
        response_ids = self._ids(response)[: self.max_response_tokens]
        before_context = self.prefix_ids + self.header_ids
        after_context = self.question_ids + question_ids + self.answer_ids + response_ids + self.instruction_ids
        budget = self.max_length - len(before_context) - len(after_context)
        context_ids = self._ids(context)
        truncated = len(context_ids) > budget
        return before_context + context_ids[:budget] + after_context, truncated

    def encode_frame(self, df: pd.DataFrame) -> tuple[Dict[str, list], EncodingStats]:
        """Encode a whole DataFrame (columns context/question/response/label)."""
        input_ids, truncated = [], 0
        for context, question, response in zip(df["context"], df["question"], df["response"]):
            ids, cut = self.encode(context, question, response)
            input_ids.append(ids)
            truncated += int(cut)
        stats = EncodingStats(
            examples=len(input_ids),
            truncated_contexts=truncated,
            mean_length=sum(map(len, input_ids)) / max(len(input_ids), 1),
        )
        columns = {
            "input_ids": input_ids,
            "attention_mask": [[1] * len(ids) for ids in input_ids],
            "labels": df["label"].astype(int).tolist(),
        }
        return columns, stats

    def to_dataset(self, df: pd.DataFrame):
        """Encode ``df`` into a Hugging Face ``datasets.Dataset`` ready for the Trainer."""
        from datasets import Dataset  # lazy import keeps unit tests light

        columns, stats = self.encode_frame(df)
        return Dataset.from_dict(columns), stats
