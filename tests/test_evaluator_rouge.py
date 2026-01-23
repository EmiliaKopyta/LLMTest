from __future__ import annotations
import json
from pathlib import Path
import pytest
from lib.evaluators.evaluate_rouge import (
    normalize,
    load_jsonl,
    compute_rouge_scores,
    summarize_rouge,
    collect_rouge_mismatches,
    evaluate_rouge,
)

# Helpers

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _write_text(path: Path, text: str) -> None:
    """Write raw text to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

# Unit tests: normalize

def test_normalize_returns_empty_for_none():
    """Verify normalize returns empty string for None."""
    assert normalize(None) == ""

def test_normalize_strips_whitespace():
    """Verify normalize strips whitespace."""
    assert normalize("  hello  ") == "hello"

# Unit tests: load jsonl file

def test_load_jsonl_raises_if_missing(tmp_path: Path):
    """Verify load_jsonl raises FileNotFoundError when file does not exist."""
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        load_jsonl(str(missing))

def test_load_jsonl_raises_on_invalid_json(tmp_path: Path):
    """Verify load_jsonl raises ValueError for invalid JSON line."""
    p = tmp_path / "bad.jsonl"
    _write_text(p, "{not valid json}\n")
    with pytest.raises(ValueError):
        load_jsonl(str(p))

def test_load_jsonl_raises_on_empty_file(tmp_path: Path):
    """Verify load_jsonl raises ValueError when file is empty."""
    p = tmp_path / "empty.jsonl"
    _write_text(p, "")
    with pytest.raises(ValueError):
        load_jsonl(str(p))

def test_load_jsonl_reads_valid_lines(tmp_path: Path):
    """Verify load_jsonl reads JSON objects from file."""
    p = tmp_path / "ok.jsonl"
    _write_jsonl(p, [{"a": 1}, {"b": 2}])

    data = load_jsonl(str(p))

    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["a"] == 1
    assert data[1]["b"] == 2

# Unit tests: compute_rouge_scores and summarize

def test_compute_rouge_scores_returns_three_lists_same_length():
    """Verify compute_rouge_scores returns 3 score lists with equal length."""
    data = [
        {"summary": "the cat is on the mat", "model_answer": "cat on mat"},
        {"summary": "a quick brown fox", "model_answer": "quick brown fox"},
    ]

    r1, r2, rl = compute_rouge_scores(data)

    assert len(r1) == len(r2) == len(rl) == 2
    assert all(isinstance(x, float) for x in r1)
    assert all(isinstance(x, float) for x in r2)
    assert all(isinstance(x, float) for x in rl)

def test_compute_rouge_scores_skips_empty_pairs():
    """Verify rows with missing reference or prediction are skipped."""
    data = [
        {"summary": "", "model_answer": "x"}, #skip
        {"summary": "ref", "model_answer": ""}, #skip
        {"summary": "hello world", "model_answer": "hello"},
    ]

    r1, r2, rl = compute_rouge_scores(data)

    assert len(r1) == len(r2) == len(rl) == 1

def test_summarize_rouge_handles_empty_scores():
    """Verify summarize_rouge returns zeros when given empty lists."""
    metrics = summarize_rouge([], [], [])
    assert metrics["samples"] == 0
    assert metrics["rouge1_f1_avg"] == 0
    assert metrics["rouge2_f1_avg"] == 0
    assert metrics["rougeL_f1_avg"] == 0

def test_summarize_rouge_averages_correctly():
    """Verify summarize_rouge computes correct averages."""
    metrics = summarize_rouge([1.0, 0.0], [0.5, 0.5], [0.2, 0.6])

    assert metrics["samples"] == 2
    assert metrics["rouge1_f1_avg"] == 0.5
    assert metrics["rouge2_f1_avg"] == 0.5
    assert metrics["rougeL_f1_avg"] == 0.4

# Unit tests: mismatches

def test_collect_rouge_mismatches_sorts_by_rougel_and_limits():
    """Verify mismatches are sorted by rougeL_f1 ascending and sliced."""
    data = [
        {"summary": "ref1", "model_answer": "pred1"},
        {"summary": "ref2", "model_answer": "pred2"},
        {"summary": "ref3", "model_answer": "pred3"},
    ]
    rouge1 = [0.2, 0.1, 0.3]
    rouge2 = [0.2, 0.1, 0.3]
    rougel = [0.9, 0.2, 0.5]

    out = collect_rouge_mismatches(
        data=data,
        reference_column="summary",
        model_answer_column="model_answer",
        rouge1_f1_scores=rouge1,
        rouge2_f1_scores=rouge2,
        rougel_f1_scores=rougel,
        max_mismatches=2,
    )

    assert len(out) == 2
    assert out[0]["rougeL_f1"] == 0.2
    assert out[1]["rougeL_f1"] == 0.5

def test_collect_rouge_mismatches_raises_for_negative_limit():
    """Verify negative mismatch limit raises ValueError."""
    with pytest.raises(ValueError):
        collect_rouge_mismatches(
            data=[{"summary": "x", "model_answer": "y"}],
            reference_column="summary",
            model_answer_column="model_answer",
            rouge1_f1_scores=[0.1],
            rouge2_f1_scores=[0.1],
            rougel_f1_scores=[0.1],
            max_mismatches=-1,
        )

def test_collect_rouge_mismatches_raises_on_length_mismatch():
    """Verify mismatch in list lengths raises ValueError."""
    with pytest.raises(ValueError):
        collect_rouge_mismatches(
            data=[{"summary": "x", "model_answer": "y"}],
            reference_column="summary",
            model_answer_column="model_answer",
            rouge1_f1_scores=[0.1, 0.2],
            rouge2_f1_scores=[0.1],
            rougel_f1_scores=[0.1],
            max_mismatches=2,
        )

# Offline test: evaluate_rouge

def test_evaluate_rouge_end_to_end(tmp_path: Path):
    """
    Verify evaluate_rouge runs end-to-end:
    - reads JSONL
    - computes ROUGE
    - returns metrics/per_sample/mismatches
    """
    input_path = tmp_path / "input.jsonl"
    _write_jsonl(
        input_path,
        [
            {"summary": "the cat is on the mat", "model_answer": "cat on mat"},
            {"summary": "quick brown fox", "model_answer": "quick brown fox"},
        ],
    )

    out = evaluate_rouge(
        input_path=str(input_path),
        reference_column="summary",
        model_answer_column="model_answer",
        max_mismatches=1,
        output_path=None,
    )

    assert "metrics" in out
    assert "per_sample" in out
    assert "mismatches" in out

    assert out["metrics"]["samples"] == 2
    assert len(out["per_sample"]) == 2
    assert len(out["mismatches"]) == 1

    row0 = out["per_sample"][0]
    assert "sample_id" in row0
    assert "rouge1_f1" in row0
    assert "rouge2_f1" in row0
    assert "rougeL_f1" in row0
    assert "rouge_reference" in row0
    assert "rouge_predicted" in row0

def test_evaluate_rouge_writes_output_jsonl(tmp_path: Path):
    """Verify metrics JSONL is written when output_path is provided."""
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "report.jsonl"

    _write_jsonl(
        input_path,
        [
            {"summary": "hello world", "model_answer": "hello"},
            {"summary": "abc def", "model_answer": "abc def"},
        ],
    )

    out = evaluate_rouge(
        input_path=str(input_path),
        reference_column="summary",
        model_answer_column="model_answer",
        output_path=str(output_path),
    )

    assert output_path.exists()

    with open(output_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
        loaded = json.loads(line)

    assert loaded["samples"] == out["metrics"]["samples"]
    assert "rougeL_f1_avg" in loaded
