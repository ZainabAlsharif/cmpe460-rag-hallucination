"""Architecture diagram of the Person 3 LLM judge (for the Methodology section).

Usage (from the repository root; needs only matplotlib)::

    python -m src.person3_zainab.figures

Writes ``report/figures/person3/architecture.pdf`` (vector, use this in LaTeX) and ``.png``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# Colours (reference data-viz palette, light mode) and text inks.
INK = "#0b0b0b"
INK_2 = "#52514e"
FROZEN_FILL, FROZEN_EDGE = "#ecebe8", "#8d8c87"  # frozen parts: neutral grey
TRAIN_FILL, TRAIN_EDGE = "#fbe1d6", "#eb6834"  # trained parts: palette slot 2 (orange)
DATA_FILL, DATA_EDGE = "#dceafa", "#2a78d6"  # data / inputs / outputs: palette slot 1 (blue)


class ArchitectureDiagram:
    """Draws the judge pipeline: prompt → 4-bit LLM with LoRA → Yes/No read-out → decision."""

    def __init__(self, width: float = 11.0, height: float = 5.6):
        self.fig, self.ax = plt.subplots(figsize=(width, height))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis("off")

    # ---------------------------------------------------------------- helpers
    def box(self, x, y, w, h, text="", fill=FROZEN_FILL, edge=FROZEN_EDGE, size=9, weight="normal",
            ls="-", align="center", color=INK, lw=1.3):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                         facecolor=fill, edgecolor=edge, linewidth=lw, linestyle=ls))
        if text:
            tx = x + w / 2 if align == "center" else x + 0.12
            self.ax.text(tx, y + h / 2, text, ha=align, va="center", fontsize=size, color=color,
                         weight=weight, linespacing=1.35)

    def arrow(self, x0, y0, x1, y1, color=INK_2, text=None, text_dy=0.14, text_dx=0.0):
        self.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                                          color=color, linewidth=1.3, shrinkA=0, shrinkB=0))
        if text:
            self.ax.text((x0 + x1) / 2 + text_dx, (y0 + y1) / 2 + text_dy, text,
                         ha="left" if text_dx else "center", va="center" if text_dx else "bottom",
                         fontsize=7.5, color=INK_2)

    def label(self, x, y, text, size=8, color=INK_2, ha="center", weight="normal", style="normal"):
        self.ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color, weight=weight, style=style)

    # ------------------------------------------------------------------ parts
    def draw_prompt(self):
        x, w = 0.25, 2.25
        self.label(x + w / 2, 5.25, "1. Input prompt", size=9.5, color=INK, weight="bold")
        rows = [
            ("Context: retrieved documents\n(cut first if too long)", 3.55, 0.95),
            ("Question: …", 3.0, 0.45),
            ("Answer: LLM response\nto be checked", 2.25, 0.65),
            ("Is every claim supported…?\nVerdict (Yes or No):", 1.5, 0.65),
        ]
        for text, y, h in rows:
            self.box(x, y, w, h, text, fill=DATA_FILL, edge=DATA_EDGE, size=8, align="left")
        self.label(x + w / 2, 1.2, "≤ 1,536 tokens, left-padded", size=7.5)

    def draw_llm(self):
        x, w = 3.05, 4.0
        self.label(x + w / 2, 5.25, "2. Decoder LLM (Qwen3-1.7B / Llama-3.2-1B)", size=9.5, color=INK, weight="bold")
        self.box(x, 1.05, w, 3.95, fill="#fcfcfb", edge=FROZEN_EDGE, ls="--", lw=1.0)
        self.label(x + w - 0.1, 4.82, "4-bit NF4 weights", size=7.5, ha="right", style="italic")
        # embedding (bottom) and LM head (top)
        self.box(x + 0.3, 1.25, w - 0.6, 0.42, "Token embeddings (frozen)", size=8)
        self.box(x + 0.3, 4.2, w - 0.6, 0.42, "LM head → next-token scores (frozen)", size=8)
        # one transformer layer, repeated L times
        lx, ly, lwid, lh = x + 0.3, 1.9, w - 0.6, 2.1
        self.box(lx, ly, lwid, lh, fill="#f6f5f3", edge=FROZEN_EDGE, lw=1.0)
        self.label(lx + lwid - 0.1, ly + lh - 0.17, "× L layers", size=8, ha="right", color=INK, weight="bold")
        self.box(lx + 0.15, ly + 1.12, 2.2, 0.62, "RMSNorm → causal self-attention\n(GQA, RoPE) — q, k, v, o", size=7.5)
        self.box(lx + 0.15, ly + 0.25, 2.2, 0.62, "RMSNorm → SwiGLU MLP\ngate, up, down", size=7.5)
        for yy in (ly + 1.12, ly + 0.25):
            self.box(lx + 2.5, yy + 0.08, 0.75, 0.46, "+ LoRA", fill=TRAIN_FILL, edge=TRAIN_EDGE, size=8, weight="bold")
        # vertical data flow inside the model
        cx = x + w / 2 - 0.35
        self.arrow(cx, 1.67, cx, 2.15)
        self.arrow(cx, 3.84, cx, 4.2)

    def draw_readout(self):
        x, w = 7.55, 1.6
        self.label(x + w / 2, 5.25, "3. Verbaliser read-out", size=9.5, color=INK, weight="bold")
        self.box(x, 3.95, w, 0.75, "scores at the\nlast position", size=8)
        self.box(x, 2.75, w, 0.85, "keep only\n\" Yes\"   \" No\"", fill=DATA_FILL, edge=DATA_EDGE, size=8.5)
        self.box(x, 1.55, w, 0.85, "softmax over 2\n→ P(hallucinated)", fill=DATA_FILL, edge=DATA_EDGE, size=8.5)
        self.arrow(x + w / 2, 3.95, x + w / 2, 3.6)
        self.arrow(x + w / 2, 2.75, x + w / 2, 2.4)

    def draw_decision(self):
        x, w = 9.55, 1.25
        self.label(x + w / 2, 5.25, "4. Decision", size=9.5, color=INK, weight="bold")
        self.box(x, 2.55, w, 0.8, "P ≥ τ ?\n(τ tuned on\nvalidation)", size=8)
        self.box(x, 3.8, w, 0.6, "Hallucinated", fill=DATA_FILL, edge=DATA_EDGE, size=8.5, weight="bold")
        self.box(x, 1.5, w, 0.6, "Supported", fill=DATA_FILL, edge=DATA_EDGE, size=8.5, weight="bold")
        self.arrow(x + w / 2, 3.35, x + w / 2, 3.8, text="yes", text_dy=0.0, text_dx=0.08)
        self.arrow(x + w / 2, 2.55, x + w / 2, 2.1, text="no", text_dy=0.0, text_dx=0.08)

    def draw_connections(self):
        self.arrow(2.5, 3.0, 3.05, 3.0)  # prompt → LLM
        self.arrow(7.05, 4.41, 7.55, 4.33)  # LM head → read-out
        self.arrow(9.15, 1.97, 9.55, 2.95)  # probability → decision

    def draw_lora_and_legend(self):
        y = 0.18
        self.box(3.05, y, 4.0, 0.72, "", fill="#fffaf7", edge=TRAIN_EDGE, lw=1.0)
        self.ax.text(3.17, y + 0.48, "LoRA on every projection W:", fontsize=8, color=INK, weight="bold", va="center")
        self.ax.text(3.17, y + 0.2, "h = W x + (α / r) · B A x,   W frozen;  A (r × d), B (d × r) trained,  r = 16",
                     fontsize=7.8, color=INK, va="center")
        lx = 7.55
        for i, (fill, edge, text) in enumerate([(FROZEN_FILL, FROZEN_EDGE, "frozen (not trained)"),
                                                (TRAIN_FILL, TRAIN_EDGE, "trained (< 2% of parameters)"),
                                                (DATA_FILL, DATA_EDGE, "data / prediction")]):
            yy = 0.72 - i * 0.26
            self.box(lx, yy - 0.08, 0.28, 0.16, fill=fill, edge=edge, lw=1.0)
            self.label(lx + 0.38, yy, text, size=7.5, ha="left")
        self.label(0.25, 0.55, "Loss (training): class-weighted\ncross-entropy over {Yes, No}", size=7.5, ha="left")

    def draw(self):
        self.draw_prompt()
        self.draw_llm()
        self.draw_readout()
        self.draw_decision()
        self.draw_connections()
        self.draw_lora_and_legend()
        return self.fig

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig = self.draw()
        fig.tight_layout(pad=0.2)
        fig.savefig(out_dir / "architecture.pdf")
        fig.savefig(out_dir / "architecture.png", dpi=220)
        plt.close(fig)
        print(f"Wrote {out_dir / 'architecture.pdf'} and .png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the Person 3 architecture diagram.")
    parser.add_argument("--out-dir", default="report/figures/person3")
    ArchitectureDiagram().save(Path(parser.parse_args().out_dir))


if __name__ == "__main__":
    main()
