"""Loaders that turn HaluEval-QA and FaithDial into one common format.

Every example is (context, response, label) with label 1 = hallucinated, 0 = grounded.
"""
import random
from dataclasses import dataclass, replace
from typing import Dict, List

from datasets import load_from_disk

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Example:
    context: str      # knowledge plus the question (HaluEval) or the dialogue history (FaithDial)
    response: str     # the answer that has to be judged
    knowledge: str    # the reference text alone (used by the overlap baseline)
    label: int        # 1 = hallucinated


class HaluEvalQALoader:
    """Each row gives one grounded and one hallucinated answer to the same question.

    The data has a single split, so it is divided here by knowledge passage: both answers of a question,
    and all questions that share a passage, always end up in the same split.
    """

    def __init__(self, data_dir: str, seed: int = 42, fractions=(0.7, 0.1, 0.2)):
        self.data_dir, self.seed, self.fractions = data_dir, seed, fractions

    def load(self) -> Dict[str, List[Example]]:
        rows = load_from_disk(self.data_dir)["data"]
        passages = sorted(set(rows["knowledge"]))
        random.Random(self.seed).shuffle(passages)
        n_train = int(len(passages) * self.fractions[0])
        n_valid = int(len(passages) * self.fractions[1])
        split_of = {p: "train" for p in passages[:n_train]}
        split_of.update({p: "validation" for p in passages[n_train:n_train + n_valid]})
        split_of.update({p: "test" for p in passages[n_train + n_valid:]})

        out = {s: [] for s in SPLITS}
        for row in rows:
            context = f"Knowledge: {row['knowledge']} Question: {row['question']}"
            for answer, label in ((row["right_answer"], 0), (row["hallucinated_answer"], 1)):
                out[split_of[row["knowledge"]]].append(Example(context, answer, row["knowledge"], label))
        return out


class FaithDialLoader:
    """Uses the official splits.

    For every turn whose original response is annotated as a hallucination, the original response is a
    positive example and the human-edited response is a negative example. Validation and test turns
    are annotated more than once, so identical examples are kept only once.
    """

    def __init__(self, data_dir: str, history_turns: int = 3):
        self.data_dir, self.history_turns = data_dir, history_turns

    def load(self) -> Dict[str, List[Example]]:
        data = load_from_disk(self.data_dir)
        return {split: self._build(data[split]) for split in SPLITS}

    def _build(self, rows) -> List[Example]:
        seen, out = set(), []
        for row in rows:
            original = (row["original_response"] or "").strip()
            if "Hallucination" not in (row["BEGIN"] or []) or not original:
                continue
            knowledge = row["knowledge"].strip()
            history = " ".join(row["history"][-self.history_turns:])
            context = f"Knowledge: {knowledge} Dialogue: {history}"
            for response, label in ((original, 1), (row["response"].strip(), 0)):
                key = (context, response)
                if response and key not in seen:
                    seen.add(key)
                    out.append(Example(context, response, knowledge, label))
        return out


def load_examples(dataset: str, data_dir: str, seed: int = 42, history_turns: int = 3) -> Dict[str, List[Example]]:
    if dataset == "halueval_qa":
        return HaluEvalQALoader(data_dir, seed).load()
    if dataset == "faithdial":
        return FaithDialLoader(data_dir, history_turns).load()
    raise ValueError(f"unknown dataset: {dataset!r}")


def shuffle_contexts(examples: List[Example], seed: int = 42) -> List[Example]:
    """Gives every example the context of a different random example (labels and responses unchanged).

    The labels still describe the original context. A model that checks the response against the context should
    now call most grounded responses hallucinated, while a model that ignores the context would not change.
    """
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    source = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
    return [replace(e, context=examples[source[i]].context, knowledge=examples[source[i]].knowledge)
            for i, e in enumerate(examples)]


def subsample(examples: List[Example], limit: int, seed: int = 42) -> List[Example]:
    if not limit or limit >= len(examples):
        return examples
    return random.Random(seed).sample(examples, limit)
