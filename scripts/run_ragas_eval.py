"""Run RAGAS evaluation against the committed sample eval set.

Offline smoke (network-free, uses stub_evaluate):

    python scripts/run_ragas_eval.py --stub

Real RAGAS run (server, requires LLM + embeddings endpoint):

    LLM_PROVIDER=local LOCAL_LLM_BASE_URL=http://localhost:8001/v1 \\
        python scripts/run_ragas_eval.py --eval-set data/sample/ragas_eval_set.jsonl

The real path lazily imports ragas and langchain_openai so the script remains
importable in environments where those packages are not installed.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path when run as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from invest_forge.eval.ragas_eval import RagasReport, stub_evaluate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSONL loader
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {"question", "answer", "contexts", "ground_truth"}


def load_eval_set(path: Path) -> tuple[list[str], list[str], list[list[str]], list[str]]:
    """Parse a JSONL eval set into parallel lists.

    Each line must be a JSON object with keys:
        question, answer, contexts (list of str), ground_truth.

    Args:
        path: Path to the JSONL file.

    Returns:
        ``(questions, answers, contexts, ground_truths)`` — all equal length.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If any row is missing required keys or lengths mismatch.
    """
    if not path.exists():
        raise FileNotFoundError(f"Eval set not found: {path}")

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            row: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {lineno}: invalid JSON — {exc}") from exc

        missing = _REQUIRED_KEYS - set(row.keys())
        if missing:
            raise ValueError(f"Line {lineno}: missing keys {missing}")

        if not isinstance(row["contexts"], list):
            raise ValueError(f"Line {lineno}: 'contexts' must be a list of strings")

        questions.append(str(row["question"]))
        answers.append(str(row["answer"]))
        contexts.append([str(c) for c in row["contexts"]])
        ground_truths.append(str(row["ground_truth"]))

    if not questions:
        raise ValueError(f"Eval set is empty: {path}")

    return questions, answers, contexts, ground_truths


# ---------------------------------------------------------------------------
# Threshold gate (pure helper — easily testable without ragas)
# ---------------------------------------------------------------------------


def meets_faithfulness(report: RagasReport, threshold: float) -> bool:
    """Return True when the report's faithfulness meets the threshold.

    Args:
        report: A ``RagasReport`` produced by either ``evaluate_ragas`` or
            ``stub_evaluate``.
        threshold: Minimum required faithfulness score (e.g. 0.8 for W3).

    Returns:
        True if ``report.faithfulness >= threshold``, else False.
    """
    return report.faithfulness >= threshold


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------


def _print_report(report: RagasReport, threshold: float) -> None:
    """Print the 4 RAGAS metrics and the W3 faithfulness gate result."""
    print("\n" + "=" * 60)
    print("  RAGAS Evaluation Results")
    print("=" * 60)
    print(f"  faithfulness      : {report.faithfulness:.4f}")
    print(f"  answer_relevancy  : {report.answer_relevancy:.4f}")
    print(f"  context_precision : {report.context_precision:.4f}")
    print(f"  context_recall    : {report.context_recall:.4f}")
    print("-" * 60)
    gate = meets_faithfulness(report, threshold)
    status = "PASS" if gate else "FAIL"
    print(f"  W3 gate  faithfulness >= {threshold:.2f}  →  {status}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation for InvestForge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval-set",
        default="data/sample/ragas_eval_set.jsonl",
        help="Path to the JSONL eval set.",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help=(
            "Run offline stub_evaluate instead of real RAGAS. "
            "Network-free; useful for smoke testing."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Minimum faithfulness score required to PASS (W3 gate).",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional path to write the report as JSON.",
    )
    args = parser.parse_args(argv)

    eval_path = Path(args.eval_set)
    logger.info("Loading eval set: %s", eval_path)
    questions, answers, contexts, ground_truths = load_eval_set(eval_path)
    logger.info("Loaded %d eval rows", len(questions))

    if args.stub:
        logger.info("Running stub_evaluate (offline mode)")
        report = stub_evaluate(answers, contexts)
    else:
        # Real RAGAS path — lazy imports so this script stays importable without
        # ragas/langchain installed (offline laptop environment).
        from invest_forge.common.config import get_settings
        from invest_forge.eval.ragas_eval import build_ragas_judge, evaluate_ragas

        logger.info("Building RAGAS judge from settings")
        judge_llm, judge_emb = build_ragas_judge(get_settings())
        logger.info("Running evaluate_ragas (network I/O) …")
        report = evaluate_ragas(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths,
            judge_llm=judge_llm,
            judge_embeddings=judge_emb,
        )

    _print_report(report, args.threshold)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "faithfulness": report.faithfulness,
            "answer_relevancy": report.answer_relevancy,
            "context_precision": report.context_precision,
            "context_recall": report.context_recall,
            "threshold": args.threshold,
            "pass": meets_faithfulness(report, args.threshold),
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Report written to %s", out_path)

    # Exit non-zero when the W3 faithfulness gate fails (useful for CI).
    if not meets_faithfulness(report, args.threshold):
        sys.exit(1)


if __name__ == "__main__":
    main()
