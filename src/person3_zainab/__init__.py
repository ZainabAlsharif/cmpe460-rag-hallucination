"""Person 3 (Zainab) — small LLM judges (Qwen3-1.7B, Llama-3.2-1B) fine-tuned with LoRA.

Modules
-------
config      ExperimentConfig: one YAML file in configs/person3_zainab/ per run
data        RAGBenchLoader, TriviaPlusLoader → common train/validation/test DataFrames
prompt      PromptEncoder: builds the judge prompt as token ids (truncates only the context)
model       JudgeModelFactory: 4-bit base model + LoRA adapters + 2-class head
trainer     WeightedLossTrainer: class-weighted cross-entropy (handles label imbalance)
evaluation  Predictor + compute_report: accuracy, F1, macro-F1, AUROC, confusion, per source
train       CLI: train one run and evaluate it on its test split
evaluate    CLI: evaluate a trained run on another dataset (transfer experiment)
summarize   CLI: tables (Markdown + LaTeX) and figures for the report

See README.md in this folder for the exact commands.
"""
