"""Tests for the standalone CLI (sf_hallucinate.cli)."""
from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from sf_hallucinate.cli import _build_parser, _load_batch_jsonl, _print_batch_summary, main
from tests.conftest import (
    OUTPUT_FAITHFUL,
    OUTPUT_HALLUCINATED,
    REFERENCE_PARIS,
)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_score_command_exists(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([
            "score",
            "--output", "test output text here",
            "--reference", "reference text here",
        ])
        assert args.command == "score"
        assert args.output == "test output text here"

    def test_score_file_command_exists(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([
            "score-file",
            "--output", "out.txt",
            "--reference", "ref.txt",
        ])
        assert args.command == "score-file"

    def test_batch_command_exists(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["batch", "--input", "pairs.jsonl"])
        assert args.command == "batch"

    def test_default_threshold(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["score", "--output", "x", "--reference", "y"])
        assert args.threshold == 0.5

    def test_custom_threshold(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([
            "score", "--output", "x", "--reference", "y", "--threshold", "0.3"
        ])
        assert args.threshold == pytest.approx(0.3)

    def test_default_format(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["score", "--output", "x", "--reference", "y"])
        assert args.format == "human"

    def test_json_format(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([
            "score", "--output", "x", "--reference", "y", "--format", "json"
        ])
        assert args.format == "json"

    def test_fail_on_any_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["batch", "--input", "f.jsonl", "--fail-on-any"])
        assert args.fail_on_any is True

    def test_fail_on_any_default_false(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["batch", "--input", "f.jsonl"])
        assert args.fail_on_any is False


# ---------------------------------------------------------------------------
# _load_batch_jsonl tests
# ---------------------------------------------------------------------------

class TestLoadBatchJsonl:
    def test_valid_jsonl(self, tmp_path: Path) -> None:
        content = (
            '{"output": "LLM said this.", "reference": "Source says this."}\n'
            '{"output": "Another claim.", "reference": "Another reference."}\n'
        )
        p = tmp_path / "test.jsonl"
        p.write_text(content, encoding="utf-8")
        pairs = _load_batch_jsonl(p)
        assert len(pairs) == 2
        assert pairs[0][0] == "LLM said this."
        assert pairs[0][1] == "Source says this."

    def test_invalid_json_lines_skipped(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        content = (
            '{"output": "valid.", "reference": "valid ref."}\n'
            'not json at all\n'
            '{"output": "second.", "reference": "second ref."}\n'
        )
        p = tmp_path / "test.jsonl"
        p.write_text(content, encoding="utf-8")
        pairs = _load_batch_jsonl(p)
        assert len(pairs) == 2
        captured = capsys.readouterr()
        assert "Warning" in captured.err

    def test_missing_fields_skipped(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        content = (
            '{"output": "valid.", "reference": "valid ref."}\n'
            '{"only_output": "no ref."}\n'
        )
        p = tmp_path / "test.jsonl"
        p.write_text(content, encoding="utf-8")
        pairs = _load_batch_jsonl(p)
        assert len(pairs) == 1

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        content = (
            '{"output": "valid.", "reference": "valid ref."}\n'
            '\n'
            '\n'
            '{"output": "second.", "reference": "second ref."}\n'
        )
        p = tmp_path / "test.jsonl"
        p.write_text(content, encoding="utf-8")
        pairs = _load_batch_jsonl(p)
        assert len(pairs) == 2

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        pairs = _load_batch_jsonl(p)
        assert pairs == []


# ---------------------------------------------------------------------------
# main() — score sub-command
# ---------------------------------------------------------------------------

class TestCmdScore:
    def test_faithful_output_exits_zero(self) -> None:
        # Use threshold=0.65 — faithful output risk ≈ 0.57 (paraphrased language
        # keeps TF-IDF cosine scores modest, but still below a generous threshold).
        with pytest.raises(SystemExit) as exc_info:
            main([
                "score",
                "--output", OUTPUT_FAITHFUL,
                "--reference", REFERENCE_PARIS,
                "--threshold", "0.65",
            ])
        assert exc_info.value.code == 0

    def test_hallucinated_output_exits_one(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([
                "score",
                "--output", OUTPUT_HALLUCINATED,
                "--reference", REFERENCE_PARIS,
                "--threshold", "0.1",
            ])
        assert exc_info.value.code == 1

    def test_json_output_is_valid_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            main([
                "score",
                "--output", OUTPUT_FAITHFUL,
                "--reference", REFERENCE_PARIS,
                "--format", "json",
            ])
        captured = capsys.readouterr()
        obj = json.loads(captured.out.strip())
        assert "hallucination_risk" in obj
        assert "faithfulness_score" in obj
        assert "passed" in obj

    def test_json_output_has_claim_results(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            main([
                "score",
                "--output", OUTPUT_FAITHFUL,
                "--reference", REFERENCE_PARIS,
                "--format", "json",
            ])
        captured = capsys.readouterr()
        obj = json.loads(captured.out.strip())
        assert "claim_results" in obj

    def test_empty_output_exits_one(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([
                "score",
                "--output", "   ",
                "--reference", REFERENCE_PARIS,
            ])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_empty_reference_exits_one(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([
                "score",
                "--output", OUTPUT_FAITHFUL,
                "--reference", "   ",
            ])
        assert exc_info.value.code == 1

    def test_invalid_threshold_exits(self) -> None:
        with pytest.raises(SystemExit):
            main([
                "score",
                "--output", "text",
                "--reference", "ref",
                "--threshold", "2.5",
            ])

    def test_human_format_contains_risk(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            main([
                "score",
                "--output", OUTPUT_FAITHFUL,
                "--reference", REFERENCE_PARIS,
                "--format", "human",
            ])
        captured = capsys.readouterr()
        assert "Hallucination Risk" in captured.out or "PASS" in captured.out or "FAIL" in captured.out


# ---------------------------------------------------------------------------
# main() — score-file sub-command
# ---------------------------------------------------------------------------

class TestCmdScoreFile:
    def test_valid_files_exit_zero_or_one(
        self,
        tmp_output_file: Path,
        tmp_reference_file: Path,
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([
                "score-file",
                "--output", str(tmp_output_file),
                "--reference", str(tmp_reference_file),
                "--threshold", "0.5",
            ])
        assert exc_info.value.code in (0, 1)

    def test_missing_output_file_exits_one(
        self,
        tmp_dir: Path,
        tmp_reference_file: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([
                "score-file",
                "--output", str(tmp_dir / "nonexistent.txt"),
                "--reference", str(tmp_reference_file),
            ])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_missing_reference_file_exits_one(
        self,
        tmp_output_file: Path,
        tmp_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([
                "score-file",
                "--output", str(tmp_output_file),
                "--reference", str(tmp_dir / "missing_ref.txt"),
            ])
        assert exc_info.value.code == 1

    def test_json_output_valid(
        self,
        tmp_output_file: Path,
        tmp_reference_file: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with pytest.raises(SystemExit):
            main([
                "score-file",
                "--output", str(tmp_output_file),
                "--reference", str(tmp_reference_file),
                "--format", "json",
            ])
        captured = capsys.readouterr()
        obj = json.loads(captured.out.strip())
        assert "hallucination_risk" in obj


# ---------------------------------------------------------------------------
# main() — batch sub-command
# ---------------------------------------------------------------------------

class TestCmdBatch:
    def test_valid_batch_exits(self, tmp_batch_jsonl: Path) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([
                "batch",
                "--input", str(tmp_batch_jsonl),
                "--threshold", "0.5",
            ])
        assert exc_info.value.code in (0, 1)

    def test_missing_input_file_exits_one(
        self,
        tmp_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([
                "batch",
                "--input", str(tmp_dir / "nonexistent.jsonl"),
            ])
        assert exc_info.value.code == 1

    def test_fail_on_any_exits_one_when_any_fail(self, tmp_batch_jsonl: Path) -> None:
        # tmp_batch_jsonl contains one hallucinated output → should fail with strict threshold
        with pytest.raises(SystemExit) as exc_info:
            main([
                "batch",
                "--input", str(tmp_batch_jsonl),
                "--threshold", "0.1",
                "--fail-on-any",
            ])
        assert exc_info.value.code == 1

    def test_jsonl_output_each_line_valid(
        self,
        tmp_batch_jsonl: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with pytest.raises(SystemExit):
            main([
                "batch",
                "--input", str(tmp_batch_jsonl),
                "--format", "jsonl",
            ])
        captured = capsys.readouterr()
        lines = [ln for ln in captured.out.strip().splitlines() if ln.strip()]
        assert len(lines) >= 1
        for line in lines:
            obj = json.loads(line)
            assert "hallucination_risk" in obj

    def test_empty_batch_file_exits_one(
        self,
        tmp_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        empty_file = tmp_dir / "empty.jsonl"
        empty_file.write_text("", encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            main(["batch", "--input", str(empty_file)])
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# _print_batch_summary
# ---------------------------------------------------------------------------

class TestPrintBatchSummary:
    def test_empty_results_no_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        _print_batch_summary([])
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_summary_shows_counts(
        self,
        capsys: pytest.CaptureFixture[str],
        default_scorer: "sf_hallucinate.eval.FaithfulnessScorer",  # type: ignore[name-defined] # noqa: F821
    ) -> None:
        from sf_hallucinate.eval import FaithfulnessScorer
        scorer = FaithfulnessScorer()
        r1 = scorer.score(OUTPUT_FAITHFUL, REFERENCE_PARIS)
        r2 = scorer.score(OUTPUT_HALLUCINATED, REFERENCE_PARIS)
        _print_batch_summary([r1, r2])
        captured = capsys.readouterr()
        assert "2" in captured.out  # total count


# ---------------------------------------------------------------------------
# KeyboardInterrupt handling
# ---------------------------------------------------------------------------

class TestKeyboardInterrupt:
    def test_keyboard_interrupt_exits_130(self) -> None:
        with patch(
            "sf_hallucinate.cli._cmd_score",
            side_effect=KeyboardInterrupt,
        ):
            with pytest.raises(SystemExit) as exc_info:
                main([
                    "score",
                    "--output", "text here please",
                    "--reference", "reference here",
                ])
            assert exc_info.value.code == 130
