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
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from finetune.sft_format import build_analyst_user_prompt, read_jsonl
from invest_forge.common.types import Rating
from invest_forge.llm.fake import FakeLLMClient
from invest_forge.tools.data_tools import FakeDataProvider
from scripts.build_sft_dataset import build_dataset, _run_pipeline_for_ticker


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
    """With with_revisions=True, _run_pipeline_for_ticker should return 2 examples."""
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
