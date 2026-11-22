"""Local checks for Person 3 code that need no GPU, torch or downloads.

Run from the repository root:  python tests/test_person3.py
"""
import glob
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.person3_zainab.config import ExperimentConfig
from src.person3_zainab.data import RAGBenchLoader, TriviaPlusLoader, BaseLoader, _sample, DataSplits
from src.person3_zainab.prompt import PromptEncoder
from src.person3_zainab.evaluation import best_threshold, compute_report, trainer_metrics

# 1. every config parses
for path in sorted(glob.glob("configs/person3_zainab/*.yaml")):
    cfg = ExperimentConfig.from_yaml(path)
    assert cfg.eval_label_column, path
    print("config ok:", path, "| test label:", cfg.eval_label_column)
cfg = ExperimentConfig.from_yaml("configs/person3_zainab/qwen3_triviaplus_llmlabels.yaml")
assert cfg.label_column.endswith("ws_llm_aaj") and cfg.eval_label_column == "response_level_label_binary"
try:
    ExperimentConfig(run_name="x", model_name="m", dataset="ragbench")
    raise AssertionError("missing data_dir not caught")
except ValueError:
    pass

# 2. RAGBench conversion
raw = pd.DataFrame({
    "documents": [np.array(["doc a", "doc b"]), ["only"], ["x"]],
    "question": ["q1", "q2", "q3"], "response": ["r1", "r2", "r3"],
    "adherence_score": [True, False, None],
})
conv = RAGBenchLoader.convert(raw, "techqa")
assert list(conv["label"]) == [0, 1], conv
assert conv["context"].iloc[0] == "[Document 1] doc a\n\n[Document 2] doc b"
print("ragbench convert ok")

# 3. TRIVIA+ conversion incl. LLM-judge labels for train and human labels for test
tp = pd.DataFrame({
    "split": ["train", "train", "valid", "test", "test"],
    "article": ["c"] * 5, "question": ["q"] * 5, "answer": ["a", "b", "c", "d", "e"], "source": ["drop"] * 5,
    "response_level_label_binary": [0, 1, 0, 1, 0],
    "response_level_label_binary_ws_llm_aaj": [1, 1, 0, 0, None],
})
tr = TriviaPlusLoader.convert(tp, "train", "response_level_label_binary_ws_llm_aaj")
te = TriviaPlusLoader.convert(tp, "test", "response_level_label_binary")
assert list(tr["label"]) == [1, 1] and list(te["label"]) == [1, 0]
cleaned = BaseLoader._clean(TriviaPlusLoader.convert(tp, "test", "response_level_label_binary_ws_llm_aaj"))
assert len(cleaned) == 1  # NaN label dropped
print("triviaplus convert ok")

# 4. stratified sampling keeps balance
df = pd.DataFrame({"label": [0] * 900 + [1] * 100, "x": range(1000)})
s = _sample(df, 200, 42)
assert len(s) == 200 and abs(s["label"].mean() - 0.1) < 0.01
assert len(_sample(df, None, 42)) == 1000
print("sampling ok")

# 5. prompt encoder with a fake whitespace tokenizer
class FakeTok:
    def __init__(self, bos=True):
        self.bos = bos
    def __call__(self, text, add_special_tokens=False):
        ids = [hash(w) % 1000 + 10 for w in text.split()]
        return {"input_ids": ([1] if (add_special_tokens and self.bos) else []) + ids}

for bos in (True, False):
    enc = PromptEncoder(FakeTok(bos), max_length=120, max_question_tokens=10, max_response_tokens=20)
    long_ctx = " ".join(["ctx"] * 500)
    ids, cut = enc.encode(long_ctx, "what is up " * 20, "answer " * 50)
    assert len(ids) == 120 and cut, len(ids)
    assert ids[: len(enc.prefix_ids)] == enc.prefix_ids
    assert ids[-len(enc.instruction_ids):] == enc.instruction_ids  # instruction never cut
    ids2, cut2 = enc.encode("short context", "q", "a")
    assert not cut2 and len(ids2) < 120
    cols, stats = enc.encode_frame(pd.DataFrame({"context": [long_ctx, "c"], "question": ["q", "q"],
                                                 "response": ["a", "b"], "label": [1, 0]}))
    assert stats.truncated_contexts == 1 and cols["labels"] == [1, 0]
print("prompt encoder ok")

# 6. metrics
rep = compute_report([0, 0, 1, 1], [0.1, 0.6, 0.9, 0.4], ["a", "a", "b", "b"])
assert rep["accuracy"] == 0.5 and rep["confusion_matrix"] == [[1, 1], [1, 1]] and rep["auroc"] == 0.75
assert set(rep["per_source"]) == {"a", "b"}
m = trainer_metrics((np.array([[2.0, 0.0], [0.0, 2.0]]), np.array([0, 1])))
assert m["accuracy"] == 1.0 and m["f1"] == 1.0
rng = np.random.default_rng(0)
lab = rng.integers(0, 2, 400)
prob = np.clip(0.6 + 0.1 * lab + rng.normal(0, 0.02, 400), 0, 1)  # well separated but shifted above 0.5
thr = best_threshold(lab, prob)
assert 0.6 < thr < 0.75, thr  # 0.5 would call everything hallucinated; the tuned threshold must not
assert compute_report(lab, prob, threshold=thr)["macro_f1"] > 0.9
assert compute_report(lab, prob, threshold=0.5)["predicted_hallucinated_rate"] > 0.95
print("metrics + threshold ok")

# 7. summarize with fake result files
from src.person3_zainab.summarize import ResultsSummary
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for name, ds, ev, lab in [("a", "ragbench", "ragbench", "adherence_score"),
                              ("b", "ragbench", "triviaplus", "adherence_score"),
                              ("c", "triviaplus", "triviaplus", "response_level_label_binary_ws_llm_aaj"),
                              ("d", "none (zero-shot)", "triviaplus", None)]:
        (tmp / f"{name}.json").write_text(json.dumps({
            "run_name": name, "model_name": "Qwen/Qwen3-1.7B", "dataset": ds, "evaluated_on": ev,
            "train_label_column": lab, "minutes": 1,
            "test": {**rep, "per_source": rep["per_source"]}}))
    summary = ResultsSummary(tmp)
    summary.write_all(tmp / "tables", tmp / "figs")
    md = summary.markdown()
    assert "RAGBench → TRIVIA+" in md and "LLM-judge" in md and "zero-shot on TRIVIA+" in md
    assert "\\rightarrow" in (tmp / "tables" / "person3_results.tex").read_text()
    assert len(list((tmp / "figs").glob("*.png"))) == 3
print("summarize ok")
print("\nALL LOCAL TESTS PASSED")
