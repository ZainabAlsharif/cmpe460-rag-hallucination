"""Download Person 3 (Zainab)'s datasets (RAGBench subsets, DelucionQA) into data/person3_zainab/.

Usage (from the repo root):
    python data/person3_zainab/download.py
"""
import urllib.error
import urllib.request
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parent

# RAGBench subsets to use. Do NOT add "delucionqa" (it overlaps with dataset 2).
SUBSETS = ["covidqa", "finqa", "techqa", "emanual"]

DELUCIONQA_FILES = ["train.csv", "dev.csv", "test.csv"]
DELUCIONQA_URL = (
    "https://raw.githubusercontent.com/boschresearch/DelucionQA/{branch}/data/DelucionQA_final/{name}"
)


def download_ragbench() -> None:
    for subset in SUBSETS:
        print(f"  RAGBench/{subset} ...")
        ds = load_dataset("rungalileo/ragbench", subset)
        ds.save_to_disk(OUT_DIR / "ragbench" / subset)
        print(f"    {ds}")


def download_delucionqa() -> None:
    target = OUT_DIR / "delucionqa"
    target.mkdir(parents=True, exist_ok=True)
    for name in DELUCIONQA_FILES:
        for branch in ("main", "master"):
            url = DELUCIONQA_URL.format(branch=branch, name=name)
            try:
                urllib.request.urlretrieve(url, target / name)
                print(f"  {name} <- {branch}")
                break
            except urllib.error.HTTPError:
                continue
        else:
            raise RuntimeError(
                f"Could not download {name}. Clone https://github.com/boschresearch/DelucionQA "
                f"and copy data/DelucionQA_final/*.csv into {target}"
            )


def main() -> None:
    print("[1/2] RAGBench ...")
    download_ragbench()
    print("[2/2] DelucionQA ...")
    download_delucionqa()
    print("\nDone. RAGBench: datasets.load_from_disk('data/person3_zainab/ragbench/<subset>'); "
          "DelucionQA: pandas.read_csv('data/person3_zainab/delucionqa/train.csv')")


if __name__ == "__main__":
    main()
