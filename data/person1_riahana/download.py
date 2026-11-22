"""Download Person 1 (Riahana)'s datasets (HaluEval-QA, FaithDial) into data/person1_riahana/.

Usage (from the repo root):
    python data/person1_riahana/download.py
"""
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parent


def load_with_parquet_fallback(repo_id: str, config: str | None = None):
    """Load a Hub dataset; fall back to the auto-converted parquet branch if the
    repo still uses a loading script (unsupported in recent `datasets`)."""
    try:
        return load_dataset(repo_id, config)
    except Exception as err:  # noqa: BLE001
        print(f"  direct load failed ({type(err).__name__}); trying parquet branch...")
        return load_dataset(repo_id, config, revision="refs/convert/parquet")


def main() -> None:
    print("[1/2] HaluEval (qa) ...")
    halueval = load_with_parquet_fallback("pminervini/HaluEval", "qa")
    halueval.save_to_disk(OUT_DIR / "halueval_qa")
    print(f"  saved -> {OUT_DIR / 'halueval_qa'}  {halueval}")

    print("[2/2] FaithDial ...")
    faithdial = load_with_parquet_fallback("McGill-NLP/FaithDial")
    faithdial.save_to_disk(OUT_DIR / "faithdial")
    print(f"  saved -> {OUT_DIR / 'faithdial'}  {faithdial}")

    print("\nDone. Load later with: datasets.load_from_disk('data/person1_riahana/<name>')")


if __name__ == "__main__":
    main()
