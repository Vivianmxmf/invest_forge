"""Build the SFT distillation dataset for the InvestForge analyst fine-tune.

Distillation flow per ticker:
  1. Build ``GraphDeps`` (FakeDataProvider when provider=fake or no Tushare
     token is available; otherwise the composite Tushare+AKShare provider).
  2. Run ``make_data_fetcher_node`` + ``make_researcher_node`` to produce a
     real ``research_memo`` via the pipeline's normal machinery (built ONCE
     per ticker; never re-sampled during temperature augmentation).
  3. Feed the memo to the TEACHER ``LLMClient`` via ``build_analyst_user_prompt``
     (byte-identical to what ``make_analyst_node`` sends at inference time),
     sampling N times across a temperature ladder to get diverse examples.
  4. Parse and normalise each teacher JSON response; skip failures.
  5. Deduplicate byte-identical assistant outputs (deterministic teachers
     collapse to 1 unique example; real teachers at temp>0 produce variety).
  6. Optionally emit revision examples with synthetic ``risk_feedback`` lines.
  7. Shuffle with the given seed and split into train / val JSONL files.

Offline / dry-run usage (no API keys required):

    python scripts/build_sft_dataset.py \\
        --teacher-provider fake \\
        --out-dir /tmp/sft_smoke

With real keys (on server) — scale to ~100-200 examples:

    TUSHARE_TOKEN=<tok> LLM_PROVIDER=openai OPENAI_API_KEY=<key> \\
        python scripts/build_sft_dataset.py \\
        --tickers 688981.SH,600519.SH,300750.SZ,600276.SH,601398.SH \\
        --out-dir data/sft \\
        --with-revisions \\
        --samples-per-ticker 20 \\
        --min-temp 0.3 \\
        --max-temp 0.9
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path when run as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from janus_terminal.agents.nodes import (
    make_data_fetcher_node,
    make_researcher_node,
)
from janus_terminal.agents.state import empty_state
from janus_terminal.common.config import LLMConfig, get_settings
from janus_terminal.common.types import Rating
from janus_terminal.llm.client import ChatMessage, LLMClient, build_client
from janus_terminal.tools.data_tools import DataProvider, FakeDataProvider
from janus_terminal.tools.sentiment import LexiconSentiment
from finetune.sft_format import SFTExample, build_analyst_user_prompt, to_chat_example, write_jsonl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sample tickers (from data/sample/fundamentals.json)
# ---------------------------------------------------------------------------

_DEFAULT_TICKERS: list[str] = [
    "688981.SH",
    "600519.SH",
    "300750.SZ",
    "600276.SH",
    "601398.SH",
]

# Synthetic risk feedback injected for the revision-path examples.
_REVISION_FEEDBACK_TEMPLATE: str = (
    "上一轮风控意见: 评级与论据自洽度不足，风险因子未涵盖 ESG 与流动性风险，"
    "请补充并重新给出更审慎的投资观点。(ticker: {ts_code})"
)

# ---------------------------------------------------------------------------
# Normalisation helpers — mirrors make_analyst_node's normalisation logic
# ---------------------------------------------------------------------------


def _normalise_analyst_json(parsed: dict[str, Any]) -> dict[str, Any] | None:
    """Validate and normalise a teacher-generated analyst JSON dict.

    Applies the same transformations as ``make_analyst_node``:
      - ``rating``     → canonical enum value via ``Rating.parse``
      - ``confidence`` → clamped to [0, 1]
      - ``risk_factors`` → must be a list with ≥ 3 entries

    Returns the normalised dict, or ``None`` if it cannot be salvaged.
    """
    if not isinstance(parsed, dict):
        return None

    # rating
    raw_rating = parsed.get("rating")
    if not raw_rating:
        return None
    try:
        parsed = dict(parsed)
        parsed["rating"] = Rating.parse(str(raw_rating)).value
    except Exception:
        return None

    # confidence
    raw_conf = parsed.get("confidence")
    if isinstance(raw_conf, (int, float)):
        parsed["confidence"] = max(0.0, min(1.0, float(raw_conf)))
    else:
        parsed["confidence"] = 0.5

    # risk_factors — must be a list of ≥ 3 strings
    rf = parsed.get("risk_factors")
    if not isinstance(rf, list):
        return None
    if len(rf) < 3:
        return None

    return parsed


def _parse_teacher_json(text: str) -> dict[str, Any] | None:
    """Tolerant JSON parse for teacher output: strips fences, finds JSON block."""
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


# ---------------------------------------------------------------------------
# Temperature ladder + dedup helpers (pure Python, no numpy)
# ---------------------------------------------------------------------------


def _temperature_ladder(
    n: int,
    min_temp: float = 0.3,
    max_temp: float = 0.9,
) -> list[float | None]:
    """Return a list of N temperatures evenly spanning [min_temp, max_temp].

    Special case: N == 1 returns ``[None]`` so the teacher uses its default
    temperature — preserving exact backward-compatible behaviour for the
    single-sample path.

    The supported temperature range is ``[0.0, 2.0]`` (2.0 is OpenAI's
    documented ceiling; HuggingFace accepts it too).

    Args:
        n: Number of samples (>= 1).
        min_temp: Lower bound of the ladder (inclusive). Must be in [0.0, 2.0].
        max_temp: Upper bound of the ladder (inclusive). Must satisfy
            ``min_temp <= max_temp <= 2.0``.

    Returns:
        A list of length N: ``[None]`` when N==1, else N floats in
        ``[min_temp, max_temp]``.

    Raises:
        ValueError: If n < 1 or not ``0.0 <= min_temp <= max_temp <= 2.0``.
    """
    if n < 1:
        raise ValueError(f"samples_per_ticker must be >= 1; got {n}")
    if not (0.0 <= min_temp <= max_temp <= 2.0):
        raise ValueError(
            "Require 0.0 <= min_temp <= max_temp <= 2.0; "
            f"got min_temp={min_temp}, max_temp={max_temp}"
        )
    if n == 1:
        return [None]
    span = max_temp - min_temp
    return [min_temp + span * i / (n - 1) for i in range(n)]


def _dedup_examples(examples: list[SFTExample]) -> list[SFTExample]:
    """Collapse byte-identical examples by FULL identity, preserving order.

    Two examples are duplicates iff BOTH their serialised user message content
    AND their assistant message content are byte-identical — i.e. the dedup key
    is the tuple ``(user_content, assistant_content)``.  The first occurrence is
    kept; subsequent duplicates are dropped.

    Keying on the full (user, assistant) pair — not assistant content alone — is
    essential for backward compatibility: a baseline example and a revision
    example share the same memo but differ in their USER prompt (the revision
    carries the synthetic risk_feedback line).  Even if a deterministic teacher
    returns identical analyst JSON for both, they are legitimately distinct
    training prompts and must NOT collapse into one.

    This keeps fake-teacher mode honest (N identical baseline samples → 1) while
    a real teacher at temp > 0 yields genuine variety, and never erases the
    baseline-vs-revision distinction.

    Args:
        examples: Ordered list of ``SFTExample`` instances (may contain dupes).

    Returns:
        A new list with duplicates removed, preserving original order.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[SFTExample] = []
    for ex in examples:
        # Dedup key is the full (user, assistant) identity.
        user_content = next(
            m["content"] for m in ex.messages if m["role"] == "user"
        )
        assistant_content = next(
            m["content"] for m in ex.messages if m["role"] == "assistant"
        )
        key = (user_content, assistant_content)
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


# ---------------------------------------------------------------------------
# Teacher resolution
# ---------------------------------------------------------------------------


def _resolve_teacher_provider(teacher_provider: str | None) -> str:
    """Resolve the effective teacher provider string.

    Precedence: explicit ``--teacher-provider`` > ``TEACHER_LLM_PROVIDER`` env
    > the global ``LLM_PROVIDER`` (via settings).  This guarantees that an
    ``openai``/``anthropic`` teacher is honoured rather than silently falling
    back to fake.
    """
    if teacher_provider:
        return teacher_provider.lower()
    env = os.getenv("TEACHER_LLM_PROVIDER")
    if env:
        return env.lower()
    return get_settings().llm.provider.lower()


# Sensible strong-teacher defaults when ``TEACHER_LLM_MODEL`` is not set.
# "hf" targets Qwen2.5-7B-Instruct: a strong, Chinese-capable <8B instruct model
# well-suited for A-share Chinese prompts; runs in-process on the GPU server
# with no API key required.  Note that for maximum distillation quality the
# teacher should be >= the student in capability; a same-size <8B teacher is
# useful for format/self-distillation but a larger cloud model (openai/anthropic)
# will typically produce higher-quality reasoning traces.
_DEFAULT_TEACHER_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5",
    "hf": "Qwen/Qwen2.5-7B-Instruct",
    "huggingface": "Qwen/Qwen2.5-7B-Instruct",
    "transformers": "Qwen/Qwen2.5-7B-Instruct",
    "fake": "fake-llm",
}


def _resolve_teacher_model(provider: str, base: LLMConfig) -> str:
    """Pick the teacher model name without ever leaking the ``fake-llm`` default.

    Precedence: ``TEACHER_LLM_MODEL`` env > the runtime ``base.model`` *only when
    the teacher reuses the same provider* (so it is actually a valid model name
    for that API) > a provider-appropriate strong default.  This prevents the
    failure mode where ``TEACHER_LLM_PROVIDER=openai`` with no model set calls
    OpenAI with ``model="fake-llm"``.
    """
    explicit = os.getenv("TEACHER_LLM_MODEL")
    if explicit:
        return explicit
    if base.provider.lower() == provider and base.model and base.model != "fake-llm":
        return base.model
    if provider in _DEFAULT_TEACHER_MODELS:
        return _DEFAULT_TEACHER_MODELS[provider]
    raise ValueError(
        f"No teacher model for provider {provider!r}; set TEACHER_LLM_MODEL."
    )


def _build_teacher_client(provider: str) -> LLMClient:
    """Build the teacher LLM from a *dedicated* teacher config.

    Reads the ``TEACHER_*`` env vars (declared in ``.env.example``) so the
    distillation teacher is configured independently of the runtime/serving
    ``LLM_PROVIDER``.  Falls back to the matching global key when a teacher
    key is not separately supplied.  We build an explicit ``LLMConfig`` and
    pass it to ``build_client`` — we never mutate the global ``LLM_PROVIDER``.
    """
    base = get_settings().llm
    teacher_cfg = LLMConfig(
        provider=provider,
        model=_resolve_teacher_model(provider, base),
        openai_api_key=os.getenv("TEACHER_OPENAI_API_KEY") or base.openai_api_key,
        anthropic_api_key=os.getenv("TEACHER_ANTHROPIC_API_KEY") or base.anthropic_api_key,
        gemini_api_key=base.gemini_api_key,
        local_base_url=base.local_base_url,
        local_model=base.local_model,
        local_analyst_model=base.local_analyst_model,
        temperature=base.temperature,
        # Pass through HF device/dtype so --teacher-provider hf respects
        # HF_DEVICE / HF_DTYPE env vars (default "auto" each).
        hf_device=os.getenv("HF_DEVICE", "auto") or "auto",
        hf_dtype=os.getenv("HF_DTYPE", "auto") or "auto",
    )
    return build_client(teacher_cfg)


# ---------------------------------------------------------------------------
# Per-ticker distillation
# ---------------------------------------------------------------------------


def _run_pipeline_for_ticker(
    ts_code: str,
    data_provider: DataProvider,
    teacher: LLMClient,
    *,
    with_revisions: bool,
    samples_per_ticker: int = 1,
    min_temp: float = 0.3,
    max_temp: float = 0.9,
) -> list[SFTExample]:
    """Run data-fetcher + researcher nodes, then query the teacher.

    The research_memo is built ONCE per ticker.  The teacher is then sampled
    ``samples_per_ticker`` times across a temperature ladder for both the
    baseline prompt and (when ``with_revisions`` is True) the revision prompt.
    Byte-identical assistant outputs are deduplicated before returning.

    Args:
        ts_code: Ticker code to process.
        data_provider: Market-data backend (real or fake).
        teacher: Teacher LLM client.
        with_revisions: If True, also generate revision-path examples.
        samples_per_ticker: Number of teacher samples per prompt variant.
        min_temp: Lower bound of the temperature ladder (when N > 1).
        max_temp: Upper bound of the temperature ladder (when N > 1).

    Returns:
        Deduplicated list of ``SFTExample`` instances (may be empty on
        unrecoverable failure).
    """
    sentiment = LexiconSentiment()

    # Step 1: data_fetcher node → populate fundamentals, macro, news, rag
    data_fetcher = make_data_fetcher_node(data_provider, sentiment, None)
    state = empty_state()
    state["ts_code"] = ts_code
    patch = data_fetcher(state)
    if patch.get("errors"):
        logger.warning("data_fetcher errors for %s: %s", ts_code, patch["errors"])
        return []
    for k, v in patch.items():
        state[k] = v  # type: ignore[literal-required]

    # Step 2: researcher node → research_memo (built ONCE; not re-sampled)
    researcher = make_researcher_node(teacher)
    research_patch = researcher(state)
    research_memo: str = research_patch.get("research_memo", "")
    if not research_memo:
        logger.warning("empty research_memo for %s", ts_code)
        return []

    temps = _temperature_ladder(samples_per_ticker, min_temp, max_temp)
    raw_examples: list[SFTExample] = []

    # Step 3a: baseline analyst examples — one per temperature step
    user_prompt = build_analyst_user_prompt(research_memo)
    for t in temps:
        teacher_resp = teacher.complete(
            [ChatMessage(role="user", content=user_prompt)],
            temperature=t,
            response_format="json",
        )
        parsed = _parse_teacher_json(teacher_resp.text)
        if parsed is None:
            logger.warning(
                "teacher returned non-JSON for %s (baseline, temp=%s); skipping",
                ts_code, t,
            )
            continue
        normalised = _normalise_analyst_json(parsed)
        if normalised is None:
            logger.warning(
                "teacher JSON invalid/incomplete for %s (baseline, temp=%s); skipping",
                ts_code, t,
            )
            continue
        raw_examples.append(to_chat_example(research_memo, normalised))

    # Step 3b: revision-path examples — one per temperature step
    if with_revisions:
        feedback = _REVISION_FEEDBACK_TEMPLATE.format(ts_code=ts_code)
        user_prompt_rev = build_analyst_user_prompt(research_memo, risk_feedback=feedback)
        for t in temps:
            rev_resp = teacher.complete(
                [ChatMessage(role="user", content=user_prompt_rev)],
                temperature=t,
                response_format="json",
            )
            parsed_rev = _parse_teacher_json(rev_resp.text)
            if parsed_rev is None:
                logger.warning(
                    "teacher returned non-JSON for %s (revision, temp=%s); skipping",
                    ts_code, t,
                )
                continue
            normalised_rev = _normalise_analyst_json(parsed_rev)
            if normalised_rev is None:
                logger.warning(
                    "teacher JSON invalid for %s (revision, temp=%s); skipping",
                    ts_code, t,
                )
                continue
            raw_examples.append(
                to_chat_example(research_memo, normalised_rev, risk_feedback=feedback)
            )

    # Dedup: collapse byte-identical assistant content within this ticker's output
    examples = _dedup_examples(raw_examples)
    logger.info(
        "  → ticker %s: %d sampled, %d unique kept",
        ts_code, len(raw_examples), len(examples),
    )
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_dataset(
    tickers: list[str],
    out_dir: Path,
    *,
    val_frac: float = 0.15,
    seed: int = 42,
    teacher_provider: str | None = None,
    with_revisions: bool = False,
    samples_per_ticker: int = 1,
    min_temp: float = 0.3,
    max_temp: float = 0.9,
) -> tuple[int, int]:
    """Generate SFT examples and write train/val JSONL.

    Args:
        tickers: List of ts_code strings to process.
        out_dir: Output directory; will be created if absent.
        val_frac: Fraction of examples to put in the val split.
        seed: Random seed for shuffling / splitting.
        teacher_provider: LLM provider string (``fake``, ``openai``, …).
            ``None`` resolves via ``TEACHER_LLM_PROVIDER`` then the global
            ``LLM_PROVIDER`` (see ``_resolve_teacher_provider``).
        with_revisions: If True, also generate revision-path examples.
        samples_per_ticker: Number of teacher samples per prompt variant.
            Default 1 preserves backward-compatible single-sample behaviour.
        min_temp: Lower bound of the temperature ladder (inclusive).
            Only used when ``samples_per_ticker > 1``.
        max_temp: Upper bound of the temperature ladder (inclusive).
            Only used when ``samples_per_ticker > 1``.

    Returns:
        ``(n_train, n_val)`` counts.
    """
    if samples_per_ticker < 1:
        raise ValueError(f"samples_per_ticker must be >= 1; got {samples_per_ticker}")
    if not (0.0 <= min_temp <= max_temp <= 2.0):
        raise ValueError(
            "Require 0.0 <= min_temp <= max_temp <= 2.0; "
            f"got min_temp={min_temp}, max_temp={max_temp}"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- build teacher LLM (explicit config; never mutates LLM_PROVIDER) ---
    effective_provider = _resolve_teacher_provider(teacher_provider)
    teacher: LLMClient = _build_teacher_client(effective_provider)
    logger.info("Teacher LLM: %s (provider=%s)", teacher.name, effective_provider)

    # --- build data provider ---
    # Use the real composite provider only when the teacher is a real cloud
    # model AND a Tushare token is present; otherwise stay fully offline.
    settings = get_settings()
    use_real = effective_provider != "fake" and bool(settings.tushare_token)
    if use_real:
        from janus_terminal.tools.data_tools import build_provider
        data_provider: DataProvider = build_provider(prefer_real=True)
        logger.info("Data provider: %s", data_provider.name)
    else:
        data_provider = FakeDataProvider()
        logger.info("Data provider: fake (offline mode)")

    # --- collect examples ---
    all_examples: list[SFTExample] = []
    for ts_code in tickers:
        logger.info("Processing ticker: %s", ts_code)
        exs = _run_pipeline_for_ticker(
            ts_code,
            data_provider,
            teacher,
            with_revisions=with_revisions,
            samples_per_ticker=samples_per_ticker,
            min_temp=min_temp,
            max_temp=max_temp,
        )
        all_examples.extend(exs)

    if not all_examples:
        logger.error("No valid examples generated; check teacher LLM output.")
        return 0, 0

    # --- deterministic shuffle + split ---
    rng = random.Random(seed)
    rng.shuffle(all_examples)
    total = len(all_examples)
    if total < 2:
        raise ValueError(
            f"Need at least 2 examples to form a train/val split; got {total}. "
            "Add more tickers or enable --with-revisions."
        )
    # Reserve at least 1 for val, and clamp so >= 1 example always remains in
    # train (max(1, round(...)) alone could swallow the entire train split for
    # tiny datasets).
    n_val = max(1, round(total * val_frac))
    n_val = min(n_val, total - 1)
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]

    write_jsonl(out_dir / "train.jsonl", train_examples)
    write_jsonl(out_dir / "val.jsonl", val_examples)

    logger.info(
        "Dataset written to %s  (train=%d, val=%d, val_frac=%.2f)",
        out_dir,
        len(train_examples),
        len(val_examples),
        n_val / len(all_examples),
    )
    return len(train_examples), len(val_examples)


def _resolve_tickers(tickers_arg: str) -> list[str]:
    """Resolve --tickers argument: comma-list or path to a file."""
    p = Path(tickers_arg)
    if p.exists():
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
        return [ln for ln in lines if ln and not ln.startswith("#")]
    return [t.strip() for t in tickers_arg.split(",") if t.strip()]


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Distill the InvestForge analyst into SFT JSONL data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers",
        default=",".join(_DEFAULT_TICKERS),
        help="Comma-separated ts_code list or path to a file of tickers.",
    )
    parser.add_argument("--out-dir", default="data/sft", help="Output directory.")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Val split fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--teacher-provider",
        default=None,
        help="LLM provider to use as teacher. Defaults to LLM_PROVIDER env var.",
    )
    parser.add_argument(
        "--with-revisions",
        action="store_true",
        help="Also emit a second example per ticker with synthetic risk feedback.",
    )
    parser.add_argument(
        "--samples-per-ticker",
        type=int,
        default=1,
        help=(
            "Number of teacher samples per prompt variant (baseline + optional revision). "
            "N=1 uses the teacher default temperature (backward-compatible). "
            "N>1 spaces temperatures evenly across [--min-temp, --max-temp]."
        ),
    )
    parser.add_argument(
        "--min-temp",
        type=float,
        default=0.3,
        help="Lower bound of the temperature ladder (inclusive). Only used when --samples-per-ticker > 1.",
    )
    parser.add_argument(
        "--max-temp",
        type=float,
        default=0.9,
        help="Upper bound of the temperature ladder (inclusive). Only used when --samples-per-ticker > 1.",
    )
    args = parser.parse_args(argv)

    tickers = _resolve_tickers(args.tickers)
    logger.info("Tickers: %s", tickers)

    n_train, n_val = build_dataset(
        tickers=tickers,
        out_dir=Path(args.out_dir),
        val_frac=args.val_frac,
        seed=args.seed,
        teacher_provider=args.teacher_provider,
        with_revisions=args.with_revisions,
        samples_per_ticker=args.samples_per_ticker,
        min_temp=args.min_temp,
        max_temp=args.max_temp,
    )

    print("\n" + "=" * 60)
    print(f"  SFT dataset complete")
    print(f"  tickers processed : {len(tickers)}")
    print(f"  train examples    : {n_train}")
    print(f"  val   examples    : {n_val}")
    print(f"  total             : {n_train + n_val}")
    print(f"  output dir        : {args.out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
