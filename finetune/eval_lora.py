"""Evaluate the InvestForge analyst adapter vs the base model on the val set.

Usage (GPU server):

    # Compare adapter vs base
    python finetune/eval_lora.py --config configs/lora.yaml

    # Base model only (no adapter)
    python finetune/eval_lora.py --config configs/lora.yaml --base-only

ALL heavy imports (torch, transformers, peft) are inside ``main()`` so this
module can be imported on a Mac and its pure metric functions tested offline.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Pure metric functions (no torch -- fully testable offline)
# ---------------------------------------------------------------------------


def json_validity_rate(predictions: list[str]) -> float:
    """Fraction of prediction strings that parse as valid JSON."""
    if not predictions:
        return 0.0
    valid = sum(1 for p in predictions if _try_parse_json(p) is not None)
    return valid / len(predictions)


def rating_accuracy(preds: list[str], golds: list[str]) -> float:
    """Fraction of predictions where the ``rating`` field matches gold."""
    if not preds or len(preds) != len(golds):
        return 0.0
    correct = 0
    for p, g in zip(preds, golds):
        pd = _try_parse_json(p)
        gd = _try_parse_json(g)
        if pd is not None and gd is not None:
            if pd.get("rating") == gd.get("rating"):
                correct += 1
    return correct / len(preds)


def confidence_mae(preds: list[str], golds: list[str]) -> float:
    """Mean absolute error on the ``confidence`` field (NaN when no valid pairs)."""
    errors: list[float] = []
    for p, g in zip(preds, golds):
        pd = _try_parse_json(p)
        gd = _try_parse_json(g)
        if pd is None or gd is None:
            continue
        pc = pd.get("confidence")
        gc = gd.get("confidence")
        if isinstance(pc, (int, float)) and isinstance(gc, (int, float)):
            errors.append(abs(float(pc) - float(gc)))
    return float(sum(errors) / len(errors)) if errors else math.nan


def risk_factor_coverage(predictions: list[str], min_factors: int = 3) -> float:
    """Fraction of predictions that include at least ``min_factors`` risk factors."""
    if not predictions:
        return 0.0
    covered = 0
    for p in predictions:
        pd = _try_parse_json(p)
        if pd is None:
            continue
        rf = pd.get("risk_factors")
        if isinstance(rf, list) and len(rf) >= min_factors:
            covered += 1
    return covered / len(predictions)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# YAML dtype string → torch attribute name (kept torch-free at module scope).
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
    """Resolve a YAML dtype string to a torch dtype (raises on unknown)."""
    key = (name or "bfloat16").strip().lower()
    attr = _DTYPE_ALIASES.get(key)
    if attr is None:
        raise ValueError(
            f"Unsupported bnb_4bit_compute_dtype {name!r}; "
            f"expected one of {sorted(set(_DTYPE_ALIASES))}"
        )
    return getattr(torch_mod, attr)


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """Return parsed dict or None; never raises."""
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None


def _compute_metrics(preds: list[str], golds: list[str]) -> dict[str, float]:
    return {
        "json_validity_rate": json_validity_rate(preds),
        "rating_accuracy": rating_accuracy(preds, golds),
        "confidence_mae": confidence_mae(preds, golds),
        "risk_factor_coverage": risk_factor_coverage(preds),
    }


def _format_table(
    base_metrics: dict[str, float],
    adapter_metrics: dict[str, float] | None = None,
) -> str:
    keys = list(base_metrics.keys())
    header = f"{'Metric':<30}{'Base':>10}"
    if adapter_metrics is not None:
        header += f"{'Adapter':>10}{'Delta':>10}"
    lines = [header, "-" * len(header)]
    for k in keys:
        bv = base_metrics[k]
        bv_str = f"{bv:.4f}" if not math.isnan(bv) else "  n/a"
        row = f"{k:<30}{bv_str:>10}"
        if adapter_metrics is not None:
            av = adapter_metrics.get(k, math.nan)
            av_str = f"{av:.4f}" if not math.isnan(av) else "  n/a"
            if not math.isnan(bv) and not math.isnan(av):
                delta_str = f"{av - bv:+.4f}"
            else:
                delta_str = "  n/a"
            row += f"{av_str:>10}{delta_str:>10}"
        lines.append(row)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GPU inference (inside main so module is Mac-safe)
# ---------------------------------------------------------------------------


def _generate_preds(
    model: Any, tokenizer: Any, examples: list[dict[str, Any]], max_new_tokens: int
) -> list[str]:
    import torch  # type: ignore[import-untyped]

    results: list[str] = []
    model.eval()
    with torch.no_grad():
        for ex in examples:
            msgs = [m for m in ex["messages"] if m["role"] != "assistant"]
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
            new_tokens = out[0][inputs["input_ids"].shape[1] :]
            results.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return results


def main(argv: list[str] | None = None) -> None:
    """All heavy imports are deferred here so the module is importable on Mac."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate InvestForge analyst adapter vs base model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/lora.yaml")
    parser.add_argument("--adapter", default=None, help="Override adapter path.")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out-json", default=None, help="Write results to JSON file.")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _ROOT / config_path
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    adapter_dir = Path(args.adapter) if args.adapter else _ROOT / cfg["output_dir"]

    val_path = _ROOT / cfg["val_data"]
    examples: list[dict[str, Any]] = []
    with val_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    golds = [
        next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
        for ex in examples
    ]

    import torch  # type: ignore[import-untyped]
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore[import-untyped]

    compute_dtype = _resolve_torch_dtype(cfg.get("bnb_4bit_compute_dtype"), torch)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=cfg.get("bnb_4bit_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model: %s", cfg["base_model"])
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    base_preds = _generate_preds(base_model, tokenizer, examples, args.max_new_tokens)
    base_metrics = _compute_metrics(base_preds, golds)

    adapter_metrics: dict[str, float] | None = None
    if not args.base_only:
        from peft import PeftModel  # type: ignore[import-untyped]

        logger.info("Loading adapter: %s", adapter_dir)
        adapter_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        adapter_preds = _generate_preds(adapter_model, tokenizer, examples, args.max_new_tokens)
        adapter_metrics = _compute_metrics(adapter_preds, golds)

    print("\n" + "=" * 60)
    print("  InvestForge Analyst Evaluation")
    print(f"  val examples : {len(examples)}")
    print("=" * 60)
    print(_format_table(base_metrics, adapter_metrics))
    print()

    results: dict[str, Any] = {"val_examples": len(examples), "base": base_metrics}
    if adapter_metrics is not None:
        results["adapter"] = adapter_metrics
        results["delta"] = {
            k: adapter_metrics[k] - base_metrics[k]
            for k in base_metrics
            if not math.isnan(base_metrics[k])
            and not math.isnan(adapter_metrics.get(k, math.nan))
        }

    if args.out_json:
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
