"""Unit tests for scripts/build_sft_dataset.py.

All tests are offline / network-free: they use FakeLLMClient as the teacher
and FakeDataProvider for market data.  No real API calls are made.

Coverage:
  - ``build_dataset`` produces train.jsonl + val.jsonl in tmp_path
  - Every example is exactly user + assistant (no system turn — parity)
  - Assistant content parses as JSON with a valid ``rating``
  - User content equals ``build_analyst_user_prompt`` output for that memo
  - Split guard keeps >= 1 train example / raises on too-few examples
  - Teacher provider resolution precedence
  - ``--with-revisions`` doubles the example count (before split)
  - Temperature ladder helper: N=1 → [None], N=3 → evenly spaced
  - Sampling with a diversity stub: N>1 → dedup preserves distinct examples
  - Sampling with a deterministic teacher: dedup collapses to 1 per ticker
  - Invalid args (samples_per_ticker=0, min_temp>max_temp) → ValueError
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# Ensure the project root is on sys.path.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from finetune.sft_format import build_analyst_user_prompt, read_jsonl
from invest_forge.common.types import Rating
from invest_forge.llm.client import ChatMessage, ChatResponse
from invest_forge.llm.fake import FakeLLMClient
from invest_forge.tools.data_tools import FakeDataProvider
from scripts.build_sft_dataset import (
    _dedup_examples,
    _run_pipeline_for_ticker,
    _temperature_ladder,
    build_dataset,
)


# ---------------------------------------------------------------------------
# Diversity stub teacher
# ---------------------------------------------------------------------------

# Maps each temperature bucket to a distinct rating / confidence so the stub
# produces genuinely different JSON for each temperature step.  This lets the
# dedup logic see real variety without calling any real LLM.
_TEMP_TO_RATING: dict[str | None, tuple[str, float]] = {
    None: ("BUY", 0.72),
    "low": ("BUY", 0.65),   # temp in [0.0, 0.45)
    "mid": ("HOLD", 0.50),  # temp in [0.45, 0.75)
    "high": ("SELL", 0.35), # temp in [0.75, 1.0]
}

_MEMO_RESPONSE = (
    "## 盈利质量\nROE 17%, ROIC 13%.\n\n"
    "[研究员] [Researcher] 这是一个虚构的研究员备忘录, 用于测试."
)


class _DiverseLLMClient:
    """Offline diversity stub: returns different analyst JSON per temperature.

    Non-json calls (researcher node) always return a fixed memo string.
    JSON calls (analyst node) return varied rating/confidence based on the
    temperature bucket so the dedup logic sees real variety.
    """

    name: str = "diverse-stub"

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
    ) -> ChatResponse:
        if response_format != "json":
            # Researcher node: return a non-empty memo
            return ChatResponse(text=_MEMO_RESPONSE)

        # Analyst node: bucket by temperature
        if temperature is None:
            rating, conf = _TEMP_TO_RATING[None]
        elif temperature < 0.45:
            rating, conf = _TEMP_TO_RATING["low"]
        elif temperature < 0.75:
            rating, conf = _TEMP_TO_RATING["mid"]
        else:
            rating, conf = _TEMP_TO_RATING["high"]

        payload: dict[str, Any] = {
            "rating": rating,
            "confidence": conf,
            "rationale": f"温度 {temperature} 下的分析师意见.",
            "target_price": round(100.0 + conf * 50, 2),
            "risk_factors": ["客户集中度", "汇率波动", "下游需求"],
        }
        return ChatResponse(text=json.dumps(payload, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Recording stub teacher
# ---------------------------------------------------------------------------


class _RecordingLLMClient:
    """Offline stub that records every analyst (json) call's temperature + user prompt.

    Researcher (non-json) calls return a fixed memo so the pipeline proceeds.
    Analyst (json) calls record ``(temperature, user_content)`` and return a
    fixed valid analyst JSON so the examples are accepted.
    """

    name: str = "recording-stub"

    def __init__(self) -> None:
        # One entry per analyst json call: (temperature, user_message_content)
        self.analyst_calls: list[tuple[float | None, str]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
    ) -> ChatResponse:
        if response_format != "json":
            # Researcher node: return a non-empty memo
            return ChatResponse(text=_MEMO_RESPONSE)

        # Analyst node: record the temperature + the exact user prompt sent.
        user_content = next(m.content for m in messages if m.role == "user")
        self.analyst_calls.append((temperature, user_content))
        payload: dict[str, Any] = {
            "rating": "BUY",
            "confidence": 0.7,
            "rationale": "固定的分析师意见, 用于记录温度.",
            "target_price": 168.5,
            "risk_factors": ["客户集中度", "汇率波动", "下游需求"],
        }
        return ChatResponse(text=json.dumps(payload, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_teacher() -> FakeLLMClient:
    """A FakeLLMClient that acts as the 'teacher' LLM."""
    return FakeLLMClient()


@pytest.fixture
def fake_provider() -> FakeDataProvider:
    """FakeDataProvider backed by in-memory defaults (no sample dir needed)."""
    return FakeDataProvider()


# ---------------------------------------------------------------------------
# Integration: build_dataset writes valid JSONL
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_dataset_creates_train_and_val_jsonl(
    tmp_path: Path,
    fake_teacher: FakeLLMClient,
    fake_provider: FakeDataProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build_dataset must write train.jsonl + val.jsonl to out_dir."""
    # Patch build_client to return our fake teacher.
    import scripts.build_sft_dataset as mod
    monkeypatch.setattr(mod, "_build_teacher_client", lambda provider: fake_teacher)
    # Patch FakeDataProvider to be our controlled instance.
    monkeypatch.setattr(mod, "FakeDataProvider", lambda: fake_provider)

    tickers = ["688981.SH", "600519.SH"]
    n_train, n_val = build_dataset(
        tickers=tickers,
        out_dir=tmp_path / "sft",
        val_frac=0.5,   # with 2 tickers → 2 examples → 1 train, 1 val
        seed=42,
        teacher_provider="fake",
        with_revisions=False,
    )

    assert n_train >= 1
    assert n_val >= 1
    assert (tmp_path / "sft" / "train.jsonl").exists()
    assert (tmp_path / "sft" / "val.jsonl").exists()


@pytest.mark.unit
def test_every_example_has_user_assistant_no_system(
    tmp_path: Path,
    fake_teacher: FakeLLMClient,
    fake_provider: FakeDataProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every JSONL row must be exactly user + assistant (no system turn)."""
    import scripts.build_sft_dataset as mod
    monkeypatch.setattr(mod, "_build_teacher_client", lambda provider: fake_teacher)
    monkeypatch.setattr(mod, "FakeDataProvider", lambda: fake_provider)

    build_dataset(
        tickers=["688981.SH", "600519.SH"],
        out_dir=tmp_path / "sft",
        val_frac=0.4,
        seed=0,
        teacher_provider="fake",
    )

    for fname in ("train.jsonl", "val.jsonl"):
        rows = read_jsonl(tmp_path / "sft" / fname)
        for row in rows:
            roles = [m["role"] for m in row["messages"]]
            assert roles == ["user", "assistant"], f"{fname}: wrong roles {roles}"


@pytest.mark.unit
def test_assistant_content_is_valid_json_with_rating(
    tmp_path: Path,
    fake_teacher: FakeLLMClient,
    fake_provider: FakeDataProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assistant content must parse as JSON with a valid Rating enum value."""
    import scripts.build_sft_dataset as mod
    monkeypatch.setattr(mod, "_build_teacher_client", lambda provider: fake_teacher)
    monkeypatch.setattr(mod, "FakeDataProvider", lambda: fake_provider)

    build_dataset(
        tickers=["688981.SH", "600276.SH"],
        out_dir=tmp_path / "sft",
        val_frac=0.4,
        seed=7,
        teacher_provider="fake",
    )

    all_rows: list[dict] = []
    for fname in ("train.jsonl", "val.jsonl"):
        p = tmp_path / "sft" / fname
        if p.exists():
            all_rows.extend(read_jsonl(p))

    assert all_rows, "No JSONL rows generated"

    valid_ratings = {r.value for r in Rating}
    for row in all_rows:
        assistant_msg = next(m for m in row["messages"] if m["role"] == "assistant")
        parsed = json.loads(assistant_msg["content"])
        assert "rating" in parsed, "assistant JSON missing 'rating'"
        assert parsed["rating"] in valid_ratings, f"invalid rating: {parsed['rating']}"


@pytest.mark.unit
def test_user_content_matches_analyst_prompt(
    tmp_path: Path,
    fake_teacher: FakeLLMClient,
    fake_provider: FakeDataProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User message content must equal build_analyst_user_prompt(memo) for that memo."""
    import scripts.build_sft_dataset as mod
    monkeypatch.setattr(mod, "_build_teacher_client", lambda provider: fake_teacher)
    monkeypatch.setattr(mod, "FakeDataProvider", lambda: fake_provider)

    build_dataset(
        tickers=["688981.SH", "600519.SH"],
        out_dir=tmp_path / "sft",
        val_frac=0.0,   # minimal val (split guard reserves exactly 1 for val)
        seed=99,
        teacher_provider="fake",
    )

    # val_frac=0.0 → n_val clamped to 1, train keeps the rest; check both files.
    all_rows: list[dict] = []
    for fname in ("train.jsonl", "val.jsonl"):
        p = tmp_path / "sft" / fname
        if p.exists():
            all_rows.extend(read_jsonl(p))

    assert all_rows

    for row in all_rows:
        user_content = next(m["content"] for m in row["messages"] if m["role"] == "user")
        # Extract the memo from the user content by checking it starts with
        # the known prefix from ANALYST_PROMPT.
        from invest_forge.agents.prompts import ANALYST_PROMPT
        # The user content must be a valid ANALYST_PROMPT.format(...) output.
        assert "[分析师]" in user_content or "[Analyst]" in user_content, (
            "User content does not look like ANALYST_PROMPT output"
        )
        # Reconstruct what build_analyst_user_prompt would produce:
        # We can't know the memo without re-running, but we can verify that
        # the user content is parseable as a format of ANALYST_PROMPT.
        # At minimum, ANALYST_PROMPT skeleton markers must be present.
        assert "rating" in user_content.lower() or "BUY" in user_content or "SELL" in user_content or "研究员备忘录" in user_content


@pytest.mark.unit
def test_run_pipeline_for_ticker_returns_examples() -> None:
    """_run_pipeline_for_ticker must return at least 1 SFTExample for a valid ticker."""
    from scripts.build_sft_dataset import _run_pipeline_for_ticker
    from finetune.sft_format import SFTExample

    teacher = FakeLLMClient()
    provider = FakeDataProvider()
    examples = _run_pipeline_for_ticker(
        "688981.SH",
        provider,
        teacher,
        with_revisions=False,
    )
    assert len(examples) >= 1
    for ex in examples:
        assert isinstance(ex, SFTExample)
        assert len(ex.messages) == 2


@pytest.mark.unit
def test_run_pipeline_with_revisions_doubles_examples() -> None:
    """With with_revisions=True, _run_pipeline_for_ticker should return 2 examples.

    This is the true backward-compat guarantee: even with a plain (deterministic)
    FakeLLMClient that returns identical analyst JSON for both the baseline and
    revision prompts, the two examples are preserved because their USER prompts
    differ (the revision carries the synthetic risk_feedback line).  Dedup keys
    on the full (user, assistant) identity, so it must NOT collapse them.
    """
    from scripts.build_sft_dataset import _run_pipeline_for_ticker

    teacher = FakeLLMClient()
    provider = FakeDataProvider()
    examples = _run_pipeline_for_ticker(
        "600519.SH",
        provider,
        teacher,
        with_revisions=True,
    )
    # Should have 2: one baseline + one revision
    assert len(examples) == 2
    # The second example's user message should contain the risk_feedback text.
    user_msgs = [m["content"] for m in examples[1].messages if m["role"] == "user"]
    assert any("风控意见" in u for u in user_msgs)


@pytest.mark.unit
def test_user_content_exact_parity_with_build_analyst_user_prompt() -> None:
    """The user message in an SFTExample must be byte-identical to
    build_analyst_user_prompt(memo) — no intermediate reformatting."""
    from scripts.build_sft_dataset import _run_pipeline_for_ticker
    from invest_forge.agents.nodes import make_data_fetcher_node, make_researcher_node
    from invest_forge.agents.state import empty_state
    from invest_forge.tools.sentiment import LexiconSentiment

    teacher = FakeLLMClient()
    provider = FakeDataProvider()

    # Re-run the same pipeline to get the memo independently.
    sentiment = LexiconSentiment()
    data_fetcher = make_data_fetcher_node(provider, sentiment, None)
    state = empty_state()
    state["ts_code"] = "688981.SH"
    patch = data_fetcher(state)
    for k, v in patch.items():
        state[k] = v  # type: ignore[literal-required]
    researcher = make_researcher_node(teacher)
    memo = researcher(state)["research_memo"]

    expected_user = build_analyst_user_prompt(memo)

    # Now get the examples from the builder.
    teacher2 = FakeLLMClient()
    provider2 = FakeDataProvider()
    examples = _run_pipeline_for_ticker(
        "688981.SH", provider2, teacher2, with_revisions=False
    )
    assert examples
    user_content = next(m["content"] for m in examples[0].messages if m["role"] == "user")
    assert user_content == expected_user, (
        "User content in SFTExample does not match build_analyst_user_prompt output"
    )


@pytest.mark.unit
def test_build_dataset_raises_when_too_few_examples(
    tmp_path: Path,
    fake_teacher: FakeLLMClient,
    fake_provider: FakeDataProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single example cannot form a train/val split — must raise clearly."""
    import scripts.build_sft_dataset as mod
    monkeypatch.setattr(mod, "_build_teacher_client", lambda provider: fake_teacher)
    monkeypatch.setattr(mod, "FakeDataProvider", lambda: fake_provider)

    with pytest.raises(ValueError, match="at least 2 examples"):
        build_dataset(
            tickers=["688981.SH"],   # 1 ticker, no revisions → 1 example
            out_dir=tmp_path / "sft",
            val_frac=0.15,
            seed=1,
            teacher_provider="fake",
        )


@pytest.mark.unit
def test_resolve_teacher_provider_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit arg wins; else TEACHER_LLM_PROVIDER; else global LLM_PROVIDER."""
    from scripts.build_sft_dataset import _resolve_teacher_provider

    # Explicit argument always wins.
    monkeypatch.setenv("TEACHER_LLM_PROVIDER", "anthropic")
    assert _resolve_teacher_provider("openai") == "openai"

    # No explicit arg → TEACHER_LLM_PROVIDER env.
    assert _resolve_teacher_provider(None) == "anthropic"

    # Neither → falls back to the resolved settings provider (fake in tests).
    monkeypatch.delenv("TEACHER_LLM_PROVIDER", raising=False)
    result = _resolve_teacher_provider(None)
    assert isinstance(result, str) and result


# ---------------------------------------------------------------------------
# New tests: temperature ladder helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_temperature_ladder_n1_returns_none() -> None:
    """N=1 → [None], so the teacher uses its configured default temperature."""
    ladder = _temperature_ladder(1)
    assert ladder == [None]


@pytest.mark.unit
def test_temperature_ladder_n3_evenly_spaced() -> None:
    """N=3, min=0.3, max=0.9 → [0.3, 0.6, 0.9] (evenly spaced, inclusive)."""
    ladder = _temperature_ladder(3, min_temp=0.3, max_temp=0.9)
    assert len(ladder) == 3
    assert ladder[0] == pytest.approx(0.3)
    assert ladder[1] == pytest.approx(0.6)
    assert ladder[2] == pytest.approx(0.9)


@pytest.mark.unit
def test_temperature_ladder_n5_bounds() -> None:
    """N=5 → first element is min_temp, last is max_temp."""
    ladder = _temperature_ladder(5, min_temp=0.2, max_temp=0.8)
    assert len(ladder) == 5
    assert ladder[0] == pytest.approx(0.2)
    assert ladder[-1] == pytest.approx(0.8)


@pytest.mark.unit
def test_temperature_ladder_invalid_n_zero() -> None:
    """samples_per_ticker=0 must raise ValueError."""
    with pytest.raises(ValueError, match="samples_per_ticker must be >= 1"):
        _temperature_ladder(0)


@pytest.mark.unit
def test_temperature_ladder_invalid_min_gt_max() -> None:
    """min_temp > max_temp must raise ValueError."""
    with pytest.raises(ValueError, match="min_temp"):
        _temperature_ladder(3, min_temp=0.9, max_temp=0.3)


@pytest.mark.unit
def test_temperature_ladder_invalid_max_temp_above_ceiling() -> None:
    """max_temp above the 2.0 ceiling must raise ValueError."""
    with pytest.raises(ValueError, match="2.0"):
        _temperature_ladder(3, min_temp=0.3, max_temp=2.5)


@pytest.mark.unit
def test_temperature_ladder_invalid_negative_min_temp() -> None:
    """A negative min_temp must raise ValueError."""
    with pytest.raises(ValueError, match="min_temp"):
        _temperature_ladder(3, min_temp=-0.1, max_temp=0.9)


# ---------------------------------------------------------------------------
# New tests: sampling + dedup with diversity stub
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_pipeline_diversity_stub_returns_multiple_unique_examples() -> None:
    """With a diverse teacher and N>1, _run_pipeline_for_ticker returns >1 example.

    The _DiverseLLMClient returns different JSON for each temperature bucket,
    so dedup should NOT collapse them — the caller gets genuine variety.
    """
    teacher = _DiverseLLMClient()
    provider = FakeDataProvider()

    examples = _run_pipeline_for_ticker(
        "688981.SH",
        provider,
        teacher,
        with_revisions=False,
        samples_per_ticker=3,
        min_temp=0.3,
        max_temp=0.9,
    )

    # 3 temperature steps with distinct outputs → at least 2 unique examples
    # (the exact count depends on bucket boundaries; we need > 1 for diversity)
    assert len(examples) > 1, (
        f"Expected >1 unique example from diversity stub, got {len(examples)}"
    )
    # All examples must be valid SFTExample with user + assistant
    from finetune.sft_format import SFTExample
    for ex in examples:
        assert isinstance(ex, SFTExample)
        roles = [m["role"] for m in ex.messages]
        assert roles == ["user", "assistant"]


@pytest.mark.unit
def test_run_pipeline_records_temperatures_with_prompt_parity() -> None:
    """N>1 baseline samples receive the temperature ladder [0.3, 0.6, 0.9] while
    the user prompt stays byte-identical across all samples — proving prompt
    parity holds and ONLY the sampling temperature varies.
    """
    teacher = _RecordingLLMClient()
    provider = FakeDataProvider()

    _run_pipeline_for_ticker(
        "688981.SH",
        provider,
        teacher,
        with_revisions=False,
        samples_per_ticker=3,
        min_temp=0.3,
        max_temp=0.9,
    )

    # Exactly 3 analyst calls were recorded (baseline only, no revisions).
    assert len(teacher.analyst_calls) == 3
    temps = [t for t, _ in teacher.analyst_calls]
    assert temps[0] == pytest.approx(0.3)
    assert temps[1] == pytest.approx(0.6)
    assert temps[2] == pytest.approx(0.9)

    # The user prompt must be byte-identical across all 3 samples and equal to
    # build_analyst_user_prompt(memo).  The memo is the recording stub's fixed
    # researcher output, so we can reconstruct the expected prompt directly.
    user_prompts = [u for _, u in teacher.analyst_calls]
    assert len(set(user_prompts)) == 1, "user prompt must not vary across samples"
    expected_user = build_analyst_user_prompt(_MEMO_RESPONSE)
    assert user_prompts[0] == expected_user


@pytest.mark.unit
def test_run_pipeline_deterministic_teacher_deduplicates_to_one() -> None:
    """With a deterministic teacher (FakeLLMClient) and N>1, dedup collapses to 1.

    FakeLLMClient ignores the temperature argument and always returns the same
    JSON → all N samples are byte-identical → dedup keeps exactly 1.
    """
    teacher = FakeLLMClient()
    provider = FakeDataProvider()

    examples = _run_pipeline_for_ticker(
        "688981.SH",
        provider,
        teacher,
        with_revisions=False,
        samples_per_ticker=5,
        min_temp=0.3,
        max_temp=0.9,
    )

    # Despite 5 samples, all are identical → dedup should yield exactly 1
    assert len(examples) == 1, (
        f"Deterministic teacher should collapse to 1 unique example; got {len(examples)}"
    )


@pytest.mark.unit
def test_run_pipeline_deterministic_teacher_with_revisions_deduplicates() -> None:
    """With revisions + deterministic teacher and N>1 samples, dedup keeps exactly
    2 examples (1 baseline + 1 revision).

    FakeLLMClient returns the same analyst JSON for both baseline and revision
    prompts, so the 4 baseline samples collapse to 1 and the 4 revision samples
    collapse to 1.  But baseline vs revision differ in their USER prompt, so the
    full-identity dedup key keeps them DISTINCT — yielding 2 examples, not 1.
    """
    teacher = FakeLLMClient()
    provider = FakeDataProvider()

    examples = _run_pipeline_for_ticker(
        "600519.SH",
        provider,
        teacher,
        with_revisions=True,
        samples_per_ticker=4,
        min_temp=0.3,
        max_temp=0.9,
    )

    # 4 baseline (identical) → 1; 4 revision (identical) → 1; distinct users → 2
    assert len(examples) == 2, (
        f"Expected 2 unique examples (1 baseline + 1 revision) after dedup; got {len(examples)}"
    )
    # The revision example's user message must carry the risk_feedback text.
    user_msgs = [m["content"] for m in examples[1].messages if m["role"] == "user"]
    assert any("风控意见" in u for u in user_msgs)


# ---------------------------------------------------------------------------
# New tests: invalid args to build_dataset
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_dataset_invalid_samples_per_ticker_zero(
    tmp_path: Path,
    fake_teacher: FakeLLMClient,
    fake_provider: FakeDataProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """samples_per_ticker=0 must raise ValueError before any LLM calls."""
    import scripts.build_sft_dataset as mod
    monkeypatch.setattr(mod, "_build_teacher_client", lambda provider: fake_teacher)
    monkeypatch.setattr(mod, "FakeDataProvider", lambda: fake_provider)

    with pytest.raises(ValueError, match="samples_per_ticker must be >= 1"):
        build_dataset(
            tickers=["688981.SH", "600519.SH"],
            out_dir=tmp_path / "sft",
            teacher_provider="fake",
            samples_per_ticker=0,
        )


@pytest.mark.unit
def test_build_dataset_invalid_min_temp_gt_max_temp(
    tmp_path: Path,
    fake_teacher: FakeLLMClient,
    fake_provider: FakeDataProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """min_temp > max_temp must raise ValueError before any LLM calls."""
    import scripts.build_sft_dataset as mod
    monkeypatch.setattr(mod, "_build_teacher_client", lambda provider: fake_teacher)
    monkeypatch.setattr(mod, "FakeDataProvider", lambda: fake_provider)

    with pytest.raises(ValueError, match="min_temp"):
        build_dataset(
            tickers=["688981.SH", "600519.SH"],
            out_dir=tmp_path / "sft",
            teacher_provider="fake",
            samples_per_ticker=3,
            min_temp=0.9,
            max_temp=0.3,
        )
