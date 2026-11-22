"""Builds the LLM judge: a decoder LLM loaded in 4-bit, adapted with LoRA, read out as Yes / No.

Architecture (what is trained and what is frozen):

* **Backbone** - a decoder-only transformer (Qwen3-1.7B or Llama-3.2-1B) loaded with 4-bit
  NF4 quantisation (QLoRA) so it fits on a 16 GB T4. Its weights stay **frozen**.
* **LoRA adapters** - small low-rank matrices (rank ``lora_r``) added to every attention
  projection (q/k/v/o) and MLP projection (gate/up/down). Only these are **trained**.
* **Verbaliser read-out** - no new classification layer. The prompt ends with
  ``Verdict (Yes or No):`` and the model's own language-model head scores the next token.
  The two logits of the tokens " Yes" (supported) and " No" (hallucinated) are the class
  logits. Re-using the pretrained head makes training far more sample-efficient than a
  randomly initialised head, and it also gives a meaningful *zero-shot* baseline
  (the model before any fine-tuning).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import ExperimentConfig

LABEL_WORDS = (" Yes", " No")  # index 0 = supported, index 1 = hallucinated


class Verbalizer:
    """Turns the LM's next-token logits into 2-class logits (supported, hallucinated)."""

    def __init__(self, tokenizer):
        self.token_ids = []
        for word in LABEL_WORDS:
            ids = tokenizer(word, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ValueError(f"Label word {word!r} is not a single token for this tokenizer: {ids}")
            self.token_ids.append(ids[0])

    def class_logits(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Logits of " Yes" / " No" at the last position (inputs are LEFT-padded)."""
        # With left padding the real tokens must still get positions 0, 1, 2, ...
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            logits_to_keep=1,  # only the last position: avoids a (seq_len x vocab) logits tensor
            use_cache=False,
        )
        return outputs.logits[:, -1, self.token_ids].float()


class JudgeModelFactory:
    """Creates tokenizers and models for training and for inference."""

    @staticmethod
    def load_tokenizer(model_name_or_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:  # Llama 3.2 has no padding token
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # the last position is then always a real token
        return tokenizer

    @staticmethod
    def _quantization_config() -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,  # T4 GPUs have no bfloat16 support
        )

    @classmethod
    def load_base_model(cls, cfg: ExperimentConfig, tokenizer):
        if cfg.load_in_4bit:
            kwargs = dict(quantization_config=cls._quantization_config(), device_map={"": 0}, torch_dtype=torch.float16)
        else:
            kwargs = dict(torch_dtype=torch.float32)
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
        model.config.pad_token_id = tokenizer.pad_token_id
        if not cfg.load_in_4bit and torch.cuda.is_available():
            model = model.to("cuda")
        return model

    @classmethod
    def build_for_training(cls, cfg: ExperimentConfig):
        """Return (peft_model, tokenizer, verbalizer); only the LoRA adapters are trainable."""
        tokenizer = cls.load_tokenizer(cfg.model_name)
        model = cls.load_base_model(cfg, tokenizer)
        if cfg.load_in_4bit:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=cfg.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        elif cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False  # incompatible with gradient checkpointing
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
        return model, tokenizer, Verbalizer(tokenizer)

    @classmethod
    def load_for_inference(cls, cfg: ExperimentConfig, adapter_dir: Optional[str | Path] = None):
        """Load the frozen base model, optionally with a trained LoRA adapter.

        ``adapter_dir=None`` gives the *zero-shot* judge (no fine-tuning at all).
        """
        tokenizer = cls.load_tokenizer(str(adapter_dir) if adapter_dir else cfg.model_name)
        model = cls.load_base_model(cfg, tokenizer)
        if adapter_dir:
            model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.config.use_cache = False
        model.eval()
        return model, tokenizer, Verbalizer(tokenizer)

    @staticmethod
    def count_parameters(model) -> dict:
        """Trainable vs. total parameters (4-bit weights are counted as stored, i.e. packed)."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return {"trainable": int(trainable), "total": int(total), "trainable_pct": round(100 * trainable / total, 4)}
