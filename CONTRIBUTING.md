# Team workflow

## Branches
- `main` — always runnable. Merge into it through a pull request.
- `person1_riahana`, `person2_abdulrahman`, `person3_zainab` — your own working branch. Commit often, push daily.

```bash
git checkout -b person3_zainab          # first time only
git add src/person3_zainab configs/person3_zainab
git commit -m "Add DelucionQA loader"
git push -u origin person3_zainab
```
Open a pull request into `main` when something works end to end. Shared files (`src/common/`, `requirements.txt`, `results/results.md`) — tell the group before changing them.

## Where things go
| What | Where | Committed? |
|---|---|---|
| Your dataset loaders / model classes | `src/personN/` | ✅ |
| Your run configs | `configs/personN/*.yaml` | ✅ |
| Downloaded datasets | `data/personN/<dataset>/` | ❌ (git-ignored) |
| Checkpoints / LoRA adapters | your own Google Drive / Kaggle output | ❌ |
| Final numbers | `results/results.md` | ✅ |
| Plots for the report | `report/figures/` | ✅ |

## Running on Kaggle / Colab
Notebooks are only launchers — never put real code in them and never submit them:

```python
!git clone https://github.com/<OWNER>/cmpe460-rag-hallucination.git
%cd cmpe460-rag-hallucination
!git checkout person3_zainab
!pip install -q -r requirements.txt
!python data/person3_zainab/download.py
!python -m src.person3_zainab.train --config configs/person3_zainab/qwen3_ragbench.yaml   # once train.py exists
```

For private repos, clone with a GitHub personal access token stored as a Kaggle/Colab **secret** — never paste the token into a notebook cell or commit it.

## Deadlines
| Date | Goal |
|---|---|
| Nov 23 | Datasets downloading; Llama access requested |
| Nov 25 | One end-to-end run per person |
| Nov 29 | All 12 runs done |
| Dec 2 | Ablations done; `results/results.md` final |
| Dec 4 | Report + slides drafted |
| Dec 5 | Submit (one person) |
