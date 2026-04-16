"""Standalone CLI for sf-hallucinate.

Entry point: ``sf-hallucinate``  (also ``python -m sf_hallucinate``)

Commands
--------

score
    Score a single LLM output against a reference string.

score-file
    Score an LLM output file against a reference file.

batch
    Score a JSONL file of {"output": ..., "reference": ...} pairs.

Examples
--------
::

    # Inline strings
    sf-hallucinate score \\
        --output "The Eiffel Tower is in Berlin." \\
        --reference "The Eiffel Tower is in Paris, France." \\
        --threshold 0.4

    # Files
    sf-hallucinate score-file \\
        --output answer.txt \\
        --reference source.txt \\
        --threshold 0.5

    # Batch JSONL  (each line: {"output": "...", "reference": "..."})
    sf-hallucinate batch \\
        --input pairs.jsonl \\
        --threshold 0.5 \\
        --fail-on-any \\
        --format jsonl

Exit codes
----------
0   All outputs passed.
1   One or more outputs exceeded the hallucination threshold, OR a fatal
    error occurred (missing file, invalid JSON, etc.).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Generator, NoReturn, Sequence

from sf_hallucinate._exceptions import (
    EmptyOutputError,
    EmptyReferenceError,
    HallucinationRiskExceeded,
)
from sf_hallucinate._types import EvalConfig, ScorerResult
from sf_hallucinate.eval import FaithfulnessScorer

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_PASS_MARK = "PASS"
_FAIL_MARK = "FAIL"
_RISK_BAR_WIDTH = 30


def _risk_bar(risk: float, width: int = _RISK_BAR_WIDTH) -> str:
    filled = int(risk * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {risk:.3f}"


def _print_result_human(result: ScorerResult, *, index: int | None = None) -> None:
    prefix = f"[{index}] " if index is not None else ""
    verdict = _PASS_MARK if result.passed else _FAIL_MARK
    print(
        f"\n{prefix}Hallucination Risk  {_risk_bar(result.hallucination_risk)}  "
        f"→ {verdict}  (threshold={result.threshold})"
    )
    print(
        f"    Faithfulness score : {result.faithfulness_score:.3f}"
    )
    print(
        f"    Grounded claims    : {result.grounded_claim_count} / "
        f"{result.total_claim_count}  "
        f"({result.grounding_rate:.0%})"
    )
    if result.claim_results:
        print("    ─── Per-claim breakdown ───")
        for i, cr in enumerate(result.claim_results, 1):
            mark = "✓" if cr.grounded else "✗"
            print(f"    {mark} [{i}] sim={cr.similarity:.3f}  claim: {cr.claim[:80]}")
            if not cr.grounded and cr.best_match:
                print(f"          best match: {cr.best_match[:80]}")


def _result_to_json(result: ScorerResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Scorer factory
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _quiet_stdout() -> Generator[None, None, None]:
    """Redirect stdout to stderr for the duration.

    Used in JSON / JSONL output modes to prevent SpanForge event pretty-printing
    (or any other side-effect output) from polluting the machine-readable
    output stream.
    """
    old = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old


def _make_scorer(args: argparse.Namespace) -> FaithfulnessScorer:
    config = EvalConfig(
        threshold=args.threshold,
        grounding_threshold=getattr(args, "grounding_threshold", 0.25),
        min_claim_length=getattr(args, "min_claim_length", 15),
        fail_on_threshold=False,  # CLI handles exit code manually
        scorer_name="faithfulness",
        similarity_backend=getattr(args, "backend", "hybrid"),
        embedding_model=getattr(args, "embedding_model", "all-MiniLM-L6-v2"),
        llm_model=getattr(args, "llm_model", "gpt-4o-mini"),
        llm_api_key=getattr(args, "llm_api_key", None),
        language=getattr(args, "language", "en"),
        detect_contradictions=getattr(args, "detect_contradictions", True),
    )
    return FaithfulnessScorer(config)


# ---------------------------------------------------------------------------
# Sub-command: score
# ---------------------------------------------------------------------------

def _cmd_score(args: argparse.Namespace) -> int:
    scorer = _make_scorer(args)
    ctx: Any = _quiet_stdout() if args.format in ("json", "jsonl") else contextlib.nullcontext()
    try:
        with ctx:
            result = scorer.score(args.output, args.reference)
    except (EmptyOutputError, EmptyReferenceError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _emit_result(result, fmt=args.format, index=None)
    return 0 if result.passed else 1


# ---------------------------------------------------------------------------
# Sub-command: score-file
# ---------------------------------------------------------------------------

def _cmd_score_file(args: argparse.Namespace) -> int:
    output_path = Path(args.output)
    reference_path = Path(args.reference)

    if not output_path.exists():
        print(f"Error: output file not found: {output_path}", file=sys.stderr)
        return 1
    if not reference_path.exists():
        print(f"Error: reference file not found: {reference_path}", file=sys.stderr)
        return 1

    output_text = output_path.read_text(encoding="utf-8")
    reference_text = reference_path.read_text(encoding="utf-8")

    scorer = _make_scorer(args)
    ctx: Any = _quiet_stdout() if args.format in ("json", "jsonl") else contextlib.nullcontext()
    try:
        with ctx:
            result = scorer.score(output_text, reference_text)
    except (EmptyOutputError, EmptyReferenceError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _emit_result(result, fmt=args.format, index=None)
    return 0 if result.passed else 1


# ---------------------------------------------------------------------------
# Sub-command: batch
# ---------------------------------------------------------------------------

def _cmd_batch(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    pairs = _load_batch_jsonl(input_path)
    if not pairs:
        print("Error: input file is empty or contains no valid JSON lines.", file=sys.stderr)
        return 1

    scorer = _make_scorer(args)
    all_passed = True
    results: list[ScorerResult] = []
    use_quiet = args.format in ("json", "jsonl")

    for i, (output, reference) in enumerate(pairs, 1):
        ctx: Any = _quiet_stdout() if use_quiet else contextlib.nullcontext()
        try:
            with ctx:
                result = scorer.score(output, reference)
        except (EmptyOutputError, EmptyReferenceError) as exc:
            print(f"  [{i}] Skipped — {exc}", file=sys.stderr)
            continue

        results.append(result)
        if not result.passed:
            all_passed = False
        _emit_result(result, fmt=args.format, index=i)

    if args.format == "human":
        _print_batch_summary(results)

    if args.fail_on_any and not all_passed:
        return 1
    return 0


def _load_batch_jsonl(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj: dict[str, Any] = json.loads(raw)
            output = str(obj.get("output", "")).strip()
            reference = str(obj.get("reference", "")).strip()
            if not output or not reference:
                print(
                    f"Warning: line {lineno} missing 'output' or 'reference' — skipped.",
                    file=sys.stderr,
                )
                continue
            pairs.append((output, reference))
        except json.JSONDecodeError as exc:
            print(f"Warning: line {lineno} invalid JSON — {exc} — skipped.", file=sys.stderr)
    return pairs


def _print_batch_summary(results: list[ScorerResult]) -> None:
    if not results:
        return
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    avg_risk = sum(r.hallucination_risk for r in results) / total
    print(
        f"\n{'═' * 60}\n"
        f"Batch summary: {passed}/{total} passed  |  "
        f"avg hallucination risk: {avg_risk:.3f}"
    )


# ---------------------------------------------------------------------------
# Output dispatching
# ---------------------------------------------------------------------------

def _emit_result(
    result: ScorerResult,
    *,
    fmt: str,
    index: int | None,
) -> None:
    if fmt == "json":
        print(_result_to_json(result))
    elif fmt == "jsonl":
        print(_result_to_json(result))
    else:  # human
        _print_result_human(result, index=index)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sf-hallucinate",
        description=(
            "Faithfulness scorer for LLM outputs — "
            "built on the SpanForge EvalScorer protocol.\n\n"
            "Scores LLM outputs against reference documents for factual "
            "grounding.  Produces a hallucination risk score per output "
            "(0.0 = fully grounded, 1.0 = fully hallucinated) and fails "
            "the pipeline when the score exceeds the configured threshold."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="sf-hallucinate 1.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # ── Shared options factory ──────────────────────────────────────────────
    def _add_shared(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--threshold",
            type=float,
            default=0.5,
            metavar="FLOAT",
            help=(
                "Hallucination risk threshold in [0, 1].  The command exits "
                "with code 1 when risk exceeds this value.  Default: %(default)s"
            ),
        )
        p.add_argument(
            "--grounding-threshold",
            dest="grounding_threshold",
            type=float,
            default=0.25,
            metavar="FLOAT",
            help=(
                "Minimum hybrid similarity for a claim to be labelled grounded.  "
                "Default: %(default)s"
            ),
        )
        p.add_argument(
            "--min-claim-length",
            dest="min_claim_length",
            type=int,
            default=15,
            metavar="INT",
            help=(
                "Minimum character length for a sentence to be treated as a "
                "claim.  Default: %(default)s"
            ),
        )
        p.add_argument(
            "--format",
            choices=["human", "json", "jsonl"],
            default="human",
            help="Output format.  Default: %(default)s",
        )
        p.add_argument(
            "--backend",
            choices=["hybrid", "embedding", "llm-nli"],
            default="hybrid",
            help="Similarity backend.  Default: %(default)s",
        )
        p.add_argument(
            "--embedding-model",
            dest="embedding_model",
            default="all-MiniLM-L6-v2",
            metavar="MODEL",
            help="Sentence-transformer model (for --backend embedding).  Default: %(default)s",
        )
        p.add_argument(
            "--llm-model",
            dest="llm_model",
            default="gpt-4o-mini",
            metavar="MODEL",
            help="LLM model (for --backend llm-nli).  Default: %(default)s",
        )
        p.add_argument(
            "--llm-api-key",
            dest="llm_api_key",
            default=None,
            metavar="KEY",
            help="LLM API key (or set OPENAI_API_KEY env var).",
        )
        p.add_argument(
            "--language",
            default="en",
            metavar="LANG",
            help="ISO 639-1 language code.  Default: %(default)s",
        )
        p.add_argument(
            "--no-contradiction-detection",
            dest="detect_contradictions",
            action="store_false",
            default=True,
            help="Disable heuristic contradiction detection.",
        )

    # ── score ───────────────────────────────────────────────────────────────
    p_score = subparsers.add_parser(
        "score",
        help="Score a single output/reference pair given as CLI arguments.",
        description="Score a single LLM output against a reference string.",
    )
    p_score.add_argument(
        "--output",
        required=True,
        metavar="TEXT",
        help="LLM-generated output text to evaluate.",
    )
    p_score.add_argument(
        "--reference",
        required=True,
        metavar="TEXT",
        help="Reference document to ground claims against.",
    )
    _add_shared(p_score)
    p_score.set_defaults(func=_cmd_score)

    # ── score-file ──────────────────────────────────────────────────────────
    p_file = subparsers.add_parser(
        "score-file",
        help="Score an output file against a reference file.",
        description=(
            "Read LLM output from --output file and reference from "
            "--reference file, then score faithfulness."
        ),
    )
    p_file.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Path to file containing the LLM-generated output.",
    )
    p_file.add_argument(
        "--reference",
        required=True,
        metavar="PATH",
        help="Path to file containing the reference document.",
    )
    _add_shared(p_file)
    p_file.set_defaults(func=_cmd_score_file)

    # ── batch ───────────────────────────────────────────────────────────────
    p_batch = subparsers.add_parser(
        "batch",
        help="Score a JSONL file of output/reference pairs.",
        description=(
            'Each line in the JSONL file must be: {"output": "...", "reference": "..."}'
        ),
    )
    p_batch.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help='Path to JSONL file.  Each line: {"output": "...", "reference": "..."}',
    )
    p_batch.add_argument(
        "--fail-on-any",
        dest="fail_on_any",
        action="store_true",
        default=False,
        help=(
            "Exit with code 1 if ANY output exceeds the threshold.  "
            "Default: exit 1 only when ALL outputs exceed threshold."
        ),
    )
    _add_shared(p_batch)
    p_batch.set_defaults(func=_cmd_batch)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> NoReturn:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Validate shared numeric ranges
    if not 0.0 <= args.threshold <= 1.0:
        parser.error(f"--threshold must be in [0, 1], got {args.threshold}")
    if not 0.0 <= args.grounding_threshold <= 1.0:
        parser.error(
            f"--grounding-threshold must be in [0, 1], got {args.grounding_threshold}"
        )

    try:
        exit_code = args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)

    sys.exit(exit_code)
