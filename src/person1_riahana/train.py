"""Train and test one encoder classifier.

Usage (from the repository root):
    python -m src.person1_riahana.train --config configs/person1_riahana/roberta_halueval.yaml
    python -m src.person1_riahana.train --config ... --input-mode answer_only     # ablation
    python -m src.person1_riahana.train --config ... --max-train-examples 200     # smoke test
"""
import argparse
import time

import torch
from transformers import get_linear_schedule_with_warmup

from ..common import set_seed
from .config import INPUT_MODES, ExperimentConfig
from .datasets import load_examples, shuffle_contexts, subsample
from .evaluation import autocast, compute_report, predict_probabilities, save_results
from .model import PairEncoder, build_classifier


class Experiment:
    """data -> tokenise -> train (best epoch chosen on validation macro-F1) -> test -> save."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def prepare_data(self):
        c = self.config
        data = load_examples(c.dataset, c.data_dir, c.split_seed, c.history_turns)
        data["train"] = subsample(data["train"], c.max_train_examples, c.split_seed)
        for split in ("validation", "test"):
            data[split] = subsample(data[split], c.max_eval_examples, c.split_seed)
        if c.input_mode == "shuffled_context":
            data = {s: shuffle_contexts(v, c.split_seed) for s, v in data.items()}
        return data

    def run(self):
        c = self.config
        set_seed(c.seed)
        data = self.prepare_data()
        print({s: len(v) for s, v in data.items()}, flush=True)

        encoder = PairEncoder(c.model_name, c.max_length, c.input_mode)
        model = build_classifier(c.model_name, c.gradient_checkpointing).to(self.device)
        train_loader = encoder.loader(data["train"], c.batch_size, shuffle=True)
        valid_loader = encoder.loader(data["validation"], c.batch_size * 2)
        test_loader = encoder.loader(data["test"], c.batch_size * 2)

        best, history = self._train(model, encoder, train_loader, valid_loader, data["validation"])

        model = build_classifier(c.output_dir).to(self.device)
        prob = predict_probabilities(model, test_loader, self.device, c.amp_dtype)
        labels = [e.label for e in data["test"]]
        report = compute_report(labels, prob)
        payload = {"config": c.to_dict(), "best_epoch": best["epoch"], "history": history, "test": report}
        path = save_results(c.results_dir, c.run_name, payload, labels, prob)
        print(f"TEST acc={report['accuracy']:.4f} f1={report['f1']:.4f} macro_f1={report['macro_f1']:.4f} "
              f"auroc={report['auroc']:.4f} -> {path}", flush=True)
        return report

    def _train(self, model, encoder, train_loader, valid_loader, valid_examples):
        c = self.config
        steps_per_epoch = (len(train_loader) + c.grad_accumulation - 1) // c.grad_accumulation
        total_steps = steps_per_epoch * c.epochs
        # foreach=False updates one tensor at a time, which needs less GPU memory (matters on 4 GB cards)
        if c.eight_bit_optimizer:
            import bitsandbytes
            optimizer = bitsandbytes.optim.AdamW8bit(model.parameters(), lr=c.learning_rate,
                                                     weight_decay=c.weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay,
                                          foreach=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * c.warmup_ratio), total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=c.fp16 and self.device == "cuda")
        labels = [e.label for e in valid_examples]
        best, history = {"macro_f1": -1.0, "epoch": 0}, []

        for epoch in range(1, c.epochs + 1):
            model.train()
            start, loss_sum = time.time(), 0.0
            optimizer.zero_grad()
            for step, batch in enumerate(train_loader, 1):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with autocast(self.device, c.amp_dtype):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("loss is not finite; train this model with fp16: false")
                scaler.scale(loss / c.grad_accumulation).backward()
                loss_sum += loss.item()
                if step % c.grad_accumulation == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                if step % 100 == 0:
                    print(f"epoch {epoch} step {step}/{len(train_loader)} loss {loss_sum / step:.4f} "
                          f"({time.time() - start:.0f}s)", flush=True)

            prob = predict_probabilities(model, valid_loader, self.device, c.amp_dtype)
            report = compute_report(labels, prob)
            history.append({"epoch": epoch, "train_loss": loss_sum / len(train_loader),
                            "seconds": round(time.time() - start), "validation": report})
            print(f"epoch {epoch}: train loss {history[-1]['train_loss']:.4f} | validation "
                  f"acc {report['accuracy']:.4f} macro_f1 {report['macro_f1']:.4f}", flush=True)
            if report["macro_f1"] > best["macro_f1"]:
                best = {"macro_f1": report["macro_f1"], "epoch": epoch}
                model.save_pretrained(c.output_dir)
                encoder.tokenizer.save_pretrained(c.output_dir)
        return best, history


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-mode", choices=INPUT_MODES)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accumulation", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gradient-checkpointing", action="store_const", const=True,
                        help="slower, but needs much less GPU memory")
    parser.add_argument("--bf16", action="store_const", const=True)
    parser.add_argument("--eight-bit-optimizer", action="store_const", const=True)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-eval-examples", type=int)
    return parser.parse_args()


def main():
    args = vars(parse_args())
    config = ExperimentConfig.from_yaml(args.pop("config"), **args)
    suffix = ""
    if config.input_mode != "full":
        suffix += f"_{config.input_mode}"
    if args["seed"] is not None:
        suffix += f"_seed{config.seed}"
    if config.max_train_examples:      # smoke tests never overwrite real results
        suffix += "_debug"
        config.results_dir += "/debug"
    config.run_name += suffix
    config.output_dir += suffix
    Experiment(config).run()


if __name__ == "__main__":
    main()
