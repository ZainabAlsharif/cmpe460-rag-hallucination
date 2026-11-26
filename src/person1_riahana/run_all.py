"""Runs every Person 1 experiment one after another and skips the ones that already have results,
so it can simply be started again after an interruption.

Usage (from the repository root):
    python -m src.person1_riahana.run_all
    python -m src.person1_riahana.run_all --low-memory        # 4 GB GPU
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from .config import ExperimentConfig

CONFIG_DIR = Path("configs/person1_riahana")
DATASETS = {"halueval": "faithdial", "faithdial": "halueval"}      # dataset -> the other dataset
MODELS = ("roberta", "deberta")


def low_memory_args(model: str, dataset: str):
    """Settings that fit a 4 GB card. batch size x accumulation steps stays 16."""
    if model == "roberta":
        batch, extra = 8, []
    else:   # DeBERTa-v3 needs more memory; HaluEval has the longer inputs (up to the 384 token limit)
        batch, extra = (2 if dataset == "halueval" else 4), ["--bf16"]
    return ["--batch-size", str(batch), "--grad-accumulation", str(16 // batch), "--eight-bit-optimizer", *extra]


def build_jobs(low_memory: bool, seeds=(None,)):
    """Returns (result name, module, arguments) for every experiment, cheapest model first.

    A seed of None is the run with the seed written in the YAML file; other seeds are repetitions
    whose result names end in "_seed<number>".
    """
    jobs = []
    for model in MODELS:
        configs = {d: CONFIG_DIR / f"{model}_{d}.yaml" for d in DATASETS}
        for seed in seeds:
            seed_args = [] if seed is None else ["--seed", str(seed)]
            suffix = "" if seed is None else f"_seed{seed}"
            for mode in ("full", "answer_only"):
                for dataset, config in configs.items():
                    name = ExperimentConfig.from_yaml(config).run_name
                    extra = low_memory_args(model, dataset) if low_memory else []
                    args = ["--config", str(config), *extra, *seed_args]
                    if mode != "full":
                        name += f"_{mode}"
                        args += ["--input-mode", mode]
                    jobs.append((name + suffix, "train", args))
            for dataset, config in configs.items():
                run = ExperimentConfig.from_yaml(config)
                other = ExperimentConfig.from_yaml(configs[DATASETS[dataset]])
                args = ["--config", str(config), *seed_args]
                jobs.append((f"{run.run_name}{suffix}_wrongcontext", "evaluate", [*args, "--shuffle-context"]))
                jobs.append((f"{run.run_name}{suffix}_on_{other.dataset}", "evaluate",
                             [*args, "--data-config", str(configs[DATASETS[dataset]])]))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[],
                        help="extra repetitions with these training seeds, run after the default one")
    options = parser.parse_args()
    results = Path("results/person1_riahana")
    logs = Path("outputs/logs")
    logs.mkdir(parents=True, exist_ok=True)

    if not (results / "p1_baseline_overlap_faithdial.json").exists():
        subprocess.run([sys.executable, "-m", "src.person1_riahana.baselines"], check=True)

    failed = []
    for name, module, args in build_jobs(options.low_memory, [None, *options.seeds]):
        if (results / f"{name}.json").exists():
            print(f"skip   {name} (already done)", flush=True)
            continue
        print(f"start  {name}", flush=True)
        start = time.time()
        with open(logs / f"{name}.log", "w", encoding="utf-8") as log:
            code = subprocess.run([sys.executable, "-u", "-m", f"src.person1_riahana.{module}", *args],
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        minutes = (time.time() - start) / 60
        print(f"{'done ' if code == 0 else 'FAIL '}  {name} ({minutes:.1f} min)", flush=True)
        if code != 0:
            failed.append(name)
    print("finished; failed runs:", failed or "none", flush=True)


if __name__ == "__main__":
    main()
