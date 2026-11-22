# Data

Each person has their own folder with their two datasets:

| Folder | Dataset 1 | Dataset 2 |
|---|---|---|
| [`person1_riahana/`](person1_riahana/) | HaluEval-QA | FaithDial |
| [`person2_abdulrahman/`](person2_abdulrahman/) | RAGTruth | PsiloQA |
| [`person3_zainab/`](person3_zainab/) | RAGBench | DelucionQA |

Run `python data/personN/download.py` from the repo root. The datasets are saved inside that folder
and are **git-ignored** — the project spec forbids submitting datasets, and GitHub rejects files over 100 MB.
Only `README.md` and `download.py` in each folder are committed.
