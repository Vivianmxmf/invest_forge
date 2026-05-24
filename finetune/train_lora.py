"""QLoRA SFT trainer for the InvestForge analyst adapter.

Run on the GPU server (4× NVIDIA RTX A5000 24 GB, CUDA 12.2).  A 7B QLoRA fit
comfortably on a single A5000, and the box has 8 cards, so we train on ONE GPU
(default cuda:0).  ``device_map`` pins the whole model to that single device —
do NOT wrap this in ``accelerate launch --num_processes N`` (multi-process
launch is incompatible with a fixed single-device ``device_map``).

    # Single-GPU training (default device cuda:0)
    python finetune/train_lora.py --config configs/lora.yaml

    # Pick a specific free GPU
    python finetune/train_lora.py --config configs/lora.yaml --device 3

Dry-run (Mac / no GPU — validates config + data without importing torch):

    python finetune/train_lora.py --config configs/lora.yaml --dry-run

Prerequisites (GPU server only):
    pip install -r requirements-gpu.txt \\
        --extra-index-url https://download.pytorch.org/whl/cu121

ALL heavy imports (torch, transformers, peft, trl, bitsandbytes, datasets)
are inside ``main()`` so this module can be imported on a Mac without them.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers (no heavy imports)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]


def _load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and return the YAML config as a plain dict."""
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# Mapping of YAML dtype strings → the torch attribute name to resolve at
# runtime.  Kept as strings here so this helper stays torch-free / Mac-safe.
_DTYPE_ALIASES: dict[str, str] = {
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "float32": "float32",
    "fp32": "float32",
}


def _resolve_torch_dtype(name: str | None, torch_mod: Any) -> Any:
    """Resolve a YAML dtype string (e.g. ``"bfloat16"``) to a torch dtype.

    Raises ``ValueError`` for an unrecognised name so config typos fail loudly
    rather than silently defaulting.
    """
    key = (name or "bfloat16").strip().lower()
    attr = _DTYPE_ALIASES.get(key)
    if attr is None:
        raise ValueError(
            f"Unsupported bnb_4bit_compute_dtype {name!r}; "
            f"expected one of {sorted(set(_DTYPE_ALIASES))}"
        )
    return getattr(torch_mod, attr)


def _count_jsonl(path: str | Path) -> int:
    """Count non-empty lines in a JSONL file."""
    p = Path(path)
    if not p.exists():
        return 0
    return sum(1 for ln in p.open(encoding="utf-8") if ln.strip())


def _validate_config(cfg: dict[str, Any]) -> list[str]:
    """Return a list of validation errors (empty = OK)."""
    errors: list[str] = []
    required = [
        "base_model",
        "output_dir",
        "train_data",
        "val_data",
        "lora_r",
        "lora_alpha",
        "num_train_epochs",
        "learning_rate",
    ]
    for key in required:
        if key not in cfg:
            errors.append(f"Missing required config key: {key!r}")

    train_path = _ROOT / cfg.get("train_data", "")
    val_path = _ROOT / cfg.get("val_data", "")
    if not train_path.exists():
        errors.append(f"train_data not found: {train_path}")
    if not val_path.exists():
        errors.append(f"val_data not found: {val_path}")
    return errors


def _dry_run(cfg: dict[str, Any], device: int = 0) -> None:
    """Print resolved plan without touching torch/transformers."""
    train_n = _count_jsonl(_ROOT / cfg.get("train_data", ""))
    val_n = _count_jsonl(_ROOT / cfg.get("val_data", ""))
    # Validate the dtype string here too so a typo fails in --dry-run.
    dtype_name = (cfg.get("bnb_4bit_compute_dtype") or "bfloat16").strip().lower()
    if dtype_name not in _DTYPE_ALIASES:
        raise ValueError(
            f"Unsupported bnb_4bit_compute_dtype {cfg.get('bnb_4bit_compute_dtype')!r}; "
            f"expected one of {sorted(set(_DTYPE_ALIASES))}"
        )

    print("\n" + "=" * 70)
    print("  InvestForge LoRA — DRY RUN  (no GPU / torch needed)")
    print("=" * 70)
    print(f"  base model        : {cfg['base_model']}")
    print(f"  adapter name      : {cfg.get('adapter_name', 'investforge-analyst')}")
    print(f"  output dir        : {cfg['output_dir']}")
    print(f"  train examples    : {train_n}")
    print(f"  val   examples    : {val_n}")
    print(f"  LoRA r / alpha    : {cfg.get('lora_r')} / {cfg.get('lora_alpha')}")
    print(f"  target modules    : {cfg.get('lora_target_modules')}")
    print(f"  epochs            : {cfg.get('num_train_epochs')}")
    print(f"  per-device batch  : {cfg.get('per_device_train_batch_size')}")
    print(f"  grad accum steps  : {cfg.get('gradient_accumulation_steps')}")
    eff_batch = (
        (cfg.get("per_device_train_batch_size") or 4)
        * (cfg.get("gradient_accumulation_steps") or 1)
    )
    print(f"  effective batch   : {eff_batch}")
    print(f"  learning rate     : {cfg.get('learning_rate')}")
    print(f"  warmup ratio      : {cfg.get('warmup_ratio')}")
    print(f"  max seq len       : {cfg.get('max_seq_len')}")
    print(f"  4-bit quant type  : {cfg.get('bnb_4bit_quant_type')}")
    print(f"  double quant      : {cfg.get('bnb_4bit_double_quant')}")
    print(f"  compute dtype     : {dtype_name}")
    print(f"  bf16              : {cfg.get('bf16')}")
    print(f"  gradient ckpt     : {cfg.get('gradient_checkpointing')}")
    print(f"  device            : cuda:{device}  (single-GPU; no accelerate launch)")
    print(f"  loss masking      : completion-only (assistant JSON tokens only)")
    print(f"  seed              : {cfg.get('seed')}")
    print("=" * 70)
    print("  Config and data look valid.  Run without --dry-run on a GPU node.")
    print()


# ---------------------------------------------------------------------------
# Training (heavy imports inside — GPU server only)
# ---------------------------------------------------------------------------


def _load_hf_dataset(cfg: dict[str, Any]):  # type: ignore[return]
    """Load train + val JSONL into HuggingFace Dataset objects."""
    from datasets import Dataset  # type: ignore[import-untyped]

    def _read(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    train_rows = _read(_ROOT / cfg["train_data"])
    val_rows = _read(_ROOT / cfg["val_data"])
    return Dataset.from_list(train_rows), Dataset.from_list(val_rows)


def main(argv: list[str] | None = None) -> None:
    """Entry point — all heavy imports live here so this module is Mac-safe."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune for InvestForge analyst.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/lora.yaml",
        help="Path to the LoRA YAML config.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Single CUDA device index to train on (model is pinned here).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config + data without importing torch; then exit.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _ROOT / config_path

    cfg = _load_config(config_path)
    errors = _validate_config(cfg)

    if errors:
        for e in errors:
            logger.error(e)
        sys.exit(1)

    if args.dry_run:
        _dry_run(cfg, device=args.device)
        return

    # -----------------------------------------------------------------------
    # GPU path — heavy imports
    # -----------------------------------------------------------------------
    import torch  # type: ignore[import-untyped]
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training  # type: ignore[import-untyped]
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore[import-untyped]
    from trl import SFTConfig, SFTTrainer  # type: ignore[import-untyped]

    logger.info("torch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())

    seed = cfg.get("seed", 42)

    # --- tokeniser ---
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- QLoRA quantisation config ---
    # Honour the YAML ``bnb_4bit_compute_dtype`` string (was previously ignored).
    compute_dtype = _resolve_torch_dtype(cfg.get("bnb_4bit_compute_dtype"), torch)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=cfg.get("bnb_4bit_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # --- base model ---
    # Pin the whole model to one GPU (single-GPU QLoRA; do NOT use
    # ``accelerate launch --num_processes N`` with a fixed device_map).
    device_map = {"": args.device}
    logger.info("Pinning model to cuda:%d", args.device)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg.get("gradient_checkpointing", True),
    )

    # --- LoRA config ---
    lora_config = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("lora_target_modules"),
        bias=cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- dataset ---
    train_ds, val_ds = _load_hf_dataset(cfg)

    # Convert {"messages": [user, assistant]} into TRL's conversational
    # *prompt-completion* format. TRL >=0.18 SFTTrainer applies the chat
    # template itself and, with completion_only_loss=True (default), masks the
    # prompt so loss is computed ONLY on the assistant JSON. This replaces the
    # old manual apply_chat_template map + DataCollatorForCompletionOnlyLM,
    # which double-process and clash with TRL 0.18's built-in ChatML pipeline.
    def _to_prompt_completion(example: dict[str, Any]) -> dict[str, Any]:
        msgs = example["messages"]
        prompt = [m for m in msgs if m.get("role") != "assistant"]
        completion = [m for m in msgs if m.get("role") == "assistant"]
        return {"prompt": prompt, "completion": completion}

    train_ds = train_ds.map(
        _to_prompt_completion, remove_columns=train_ds.column_names, num_proc=1
    )
    val_ds = val_ds.map(
        _to_prompt_completion, remove_columns=val_ds.column_names, num_proc=1
    )

    # --- SFT training config ---
    output_dir = str(_ROOT / cfg["output_dir"])
    sft_cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.01),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        bf16=cfg.get("bf16", True),
        fp16=cfg.get("fp16", False),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        optim=cfg.get("optim", "paged_adamw_8bit"),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 50),
        eval_steps=cfg.get("eval_steps", 50),
        # transformers >=4.46 renamed evaluation_strategy -> eval_strategy.
        # Epoch-based eval+save keeps load_best_model_at_end valid even on tiny
        # datasets (step-based saves never trigger when total steps < save_steps).
        eval_strategy=cfg.get("eval_strategy", "epoch"),
        save_strategy=cfg.get("save_strategy", "epoch"),
        save_total_limit=cfg.get("save_total_limit", 2),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_loss"),
        report_to=cfg.get("report_to", "none"),
        seed=seed,
        dataloader_num_workers=cfg.get("dataloader_num_workers", 2),
        remove_unused_columns=cfg.get("remove_unused_columns", False),
        # TRL >=0.18 renamed max_seq_length -> max_length on SFTConfig.
        max_length=cfg.get("max_seq_len", 2048),
        # Prompt-completion dataset → mask the prompt, train on completion only.
        completion_only_loss=cfg.get("completion_only_loss", True),
        # Packing MUST stay off — it concatenates examples and is incompatible
        # with completion-only loss masking.
        packing=False,
    )

    # --- trainer ---
    # completion_only_loss (set on SFTConfig) masks the prompt so loss is on the
    # assistant JSON only (prompt tokens become -100).
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        # transformers >=4.46 / TRL >=0.18 renamed the trainer's tokenizer arg
        # to processing_class. No custom collator — TRL applies the chat
        # template and completion-only masking from the prompt-completion data.
        processing_class=tokenizer,
    )

    logger.info("Starting training …")
    trainer.train()

    # --- save adapter ---
    adapter_path = _ROOT / cfg["output_dir"]
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info("Adapter saved to %s", adapter_path)


if __name__ == "__main__":
    main()
