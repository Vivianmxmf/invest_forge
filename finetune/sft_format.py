"""SFT data format utilities for the InvestForge analyst fine-tune.

Key design principle: ``build_analyst_user_prompt`` is the *single source of
truth* for the analyst user message.  It mirrors the exact truncation and
``str.format`` call made by ``make_analyst_node`` in
``janus_terminal/agents/nodes.py`` so that train-time examples are byte-identical
to inference-time inputs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from janus_terminal.agents.prompts import ANALYST_PROMPT

# ---------------------------------------------------------------------------
# NOTE on train/serve parity: inference (``make_analyst_node``) sends EXACTLY
# one user message — ``[ChatMessage(role="user", content=prompt)]`` — with no
# system turn.  We therefore emit no system message here either; injecting one
# would make the chat template render differently at train vs serve time.  The
# analyst role instruction already lives inside ``ANALYST_PROMPT``.
# ---------------------------------------------------------------------------

# Maximum characters we feed into the ``research_memo`` slot — must stay in
# sync with the ``[:4000]`` slice in ``make_analyst_node``.
_MEMO_MAX_CHARS: int = 4000


# ---------------------------------------------------------------------------
# Core prompt builder — mirrors nodes.py make_analyst_node exactly
# ---------------------------------------------------------------------------


def build_analyst_user_prompt(
    research_memo: str,
    risk_feedback: str = "",
) -> str:
    """Return the user-turn string that the analyst node would send to the LLM.

    This is the canonical source used by *both* the dataset builder and any
    offline evaluation; changing the node logic here is the only place you
    need to touch.

    Args:
        research_memo: Raw memo text (will be truncated to 4000 chars).
        risk_feedback: Optional risk-control feedback from a prior round.

    Returns:
        The fully-formatted prompt string.
    """
    return ANALYST_PROMPT.format(
        research_memo=research_memo[:_MEMO_MAX_CHARS],
        risk_feedback=risk_feedback,
    )


# ---------------------------------------------------------------------------
# SFTExample — immutable container for one chat-format training example
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SFTExample:
    """A single supervised fine-tuning example in chat format.

    ``messages`` is a tuple of dicts, each with ``role`` and ``content`` keys,
    following the OpenAI / Hugging Face chat template convention.  For
    train/serve parity there is NO system turn — exactly user + assistant::

        [
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."},
        ]
    """

    messages: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON encoding."""
        return {"messages": list(self.messages)}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def to_chat_example(
    research_memo: str,
    gold_json: dict[str, Any],
    risk_feedback: str = "",
) -> SFTExample:
    """Build an ``SFTExample`` from a memo + gold analyst JSON.

    The assistant content is ``json.dumps(gold_json, ensure_ascii=False)`` so
    that the model learns to output compact, valid JSON.

    Args:
        research_memo: Researcher-memo text (used verbatim; truncation happens
            inside ``build_analyst_user_prompt``).
        gold_json: Teacher-generated analyst JSON dict (already validated).
        risk_feedback: Optional risk feedback string for the revision path.

    Returns:
        A frozen ``SFTExample`` with user / assistant messages (no system turn,
        to match inference parity).
    """
    return SFTExample(
        messages=(
            {"role": "user", "content": build_analyst_user_prompt(research_memo, risk_feedback)},
            {"role": "assistant", "content": json.dumps(gold_json, ensure_ascii=False)},
        )
    )


# ---------------------------------------------------------------------------
# JSONL I/O helpers
# ---------------------------------------------------------------------------


def write_jsonl(path: str | Path, examples: list[SFTExample]) -> None:
    """Write a list of ``SFTExample`` objects to a JSONL file.

    Each line is a JSON object with a ``messages`` key.  The file is written
    with UTF-8 encoding and ``ensure_ascii=False`` so CJK text is preserved.

    Args:
        path: Destination file path (parent directory must exist).
        examples: Sequence of ``SFTExample`` instances to serialise.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of parsed dicts.

    Empty lines and lines consisting solely of whitespace are skipped.

    Args:
        path: Source file path.

    Returns:
        List of dicts, one per non-empty line.
    """
    src = Path(path)
    rows: list[dict[str, Any]] = []
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
