"""Download Person 2 (Abdulrahman)'s datasets (RAGTruth, PsiloQA) into data/person2_abdulrahman/.

Usage (from the repo root):
    python data/person2_abdulrahman/download.py
"""
from pathlib import Path

from datasets import DatasetDict, load_dataset

OUT_DIR = Path(__file__).resolve().parent
LANGUAGE_COLUMNS = ("lang", "language", "locale")
ENGLISH_VALUES = {"en", "eng", "english", "en-us"}


def keep_english(ds: DatasetDict) -> DatasetDict:
    """Keep English rows if the dataset has a language column; otherwise keep everything."""
    first_split = next(iter(ds.values()))
    column = next((c for c in LANGUAGE_COLUMNS if c in first_split.column_names), None)
    if column is None:
        print(f"  no language column found (columns: {first_split.column_names}); keeping all rows.")
        return ds
    print(f"  filtering English rows using column '{column}'")
    return ds.filter(lambda row: str(row[column]).strip().lower() in ENGLISH_VALUES)


def main() -> None:
    print("[1/2] RAGTruth (processed) ...")
    ragtruth = load_dataset("wandb/RAGTruth-processed")
    ragtruth.save_to_disk(OUT_DIR / "ragtruth")
    print(f"  saved -> {OUT_DIR / 'ragtruth'}  {ragtruth}")

    print("[2/2] PsiloQA ...")
    psiloqa = keep_english(load_dataset("s-nlp/PsiloQA"))
    psiloqa.save_to_disk(OUT_DIR / "psiloqa")
    print(f"  saved -> {OUT_DIR / 'psiloqa'}  {psiloqa}")

    print("\nDone. Load later with: datasets.load_from_disk('data/person2_abdulrahman/<name>')")


if __name__ == "__main__":
    main()
