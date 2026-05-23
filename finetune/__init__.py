"""InvestForge LoRA fine-tuning package.

Submodules:
  sft_format   — SFTExample dataclass, prompt builders, jsonl I/O
  train_lora   — QLoRA SFT trainer (requires GPU + heavy deps)
  eval_lora    — Adapter vs base evaluation (requires GPU + heavy deps)
"""
from __future__ import annotations
