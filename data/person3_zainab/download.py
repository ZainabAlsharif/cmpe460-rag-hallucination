"""Download Person 3 (Zainab)'s datasets (RAGBench subsets, TRIVIA+) into data/person3_zainab/.

Usage (from the repo root):
    python data/person3_zainab/download.py
"""
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parent

# RAGBench subsets to use. covidqa is left out because TRIVIA+ also contains COVID questions;
# delucionqa is left out because it is a re-labelled copy of another dataset.
SUBSETS = ["finqa", "techqa", "emanual"]

TRIVIAPLUS_FILE = "triviaplus_dataset.parquet"
TRIVIAPLUS_URL = (
    "https://github.com/amazon-science/hallucination-benchmark-trivialplus/raw/{branch}/" + TRIVIAPLUS_FILE
)


def download_ragbench() -> None:
    for subset in SUBSETS:
        print(f"  RAGBench/{subset} ...")
        ds = load_dataset("rungalileo/ragbench", subset)
        ds.save_to_disk(OUT_DIR / "ragbench" / subset)
        print(f"    {ds}")


def download_triviaplus() -> None:
    target = OUT_DIR / "triviaplus"
    target.mkdir(parents=True, exist_ok=True)
    for branch in ("main", "master"):
        try:
            urllib.request.urlretrieve(TRIVIAPLUS_URL.format(branch=branch), target / TRIVIAPLUS_FILE)
            break
        except urllib.error.HTTPError:
            continue
    else:
        raise RuntimeError(
            "Could not download TRIVIA+. Get triviaplus_dataset.parquet from "
            f"https://github.com/amazon-science/hallucination-benchmark-trivialplus and put it in {target}"
        )
    df = pd.read_parquet(target / TRIVIAPLUS_FILE)
    print(f"    {len(df)} rows | splits: {df['split'].value_counts().to_dict()}")
    print(f"    hallucinated (human label): {df['response_level_label_binary'].mean():.1%}")


def main() -> None:
    print("[1/2] RAGBench ...")
    download_ragbench()
    print("[2/2] TRIVIA+ ...")
    download_triviaplus()
    print("\nDone. RAGBench: datasets.load_from_disk('data/person3_zainab/ragbench/<subset>'); "
          "TRIVIA+: pandas.read_parquet('data/person3_zainab/triviaplus/triviaplus_dataset.parquet')")


if __name__ == "__main__":
    main()
