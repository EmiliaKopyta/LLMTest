from __future__ import annotations
import os
import json
import pandas as pd
import pytest

from lib.evaluators.evaluate_accuracy import (
    normalize,
    strip_choice_prefix,
    parse_choices,
    load_jsonl,
    evaluate_row,
    build_per_sample_accuracy,
    collect_mismatches,
    summarize_results,
    save_report_jsonl,
    evaluate_accuracy,
)

# Helpers

def _write_jsonl(path, rows: list[dict]) -> None:
    """Write a JSONL file (records per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Unit tests: normalize

def test_normalize_handles_none():
    """Verify None becomes empty string."""
    assert normalize(None) == ""

def test_normalize_strips_quotes_and_whitespace():
    """Verify normalize removes extra whitespace and surrounding quotes."""
    assert normalize('  "Hello   world"  ') == "hello world"

def test_normalize_can_be_case_sensitive():
    """Verify ignore_case=False keeps original casing."""
    assert normalize("AbC", ignore_case=False) == "AbC"

# Unit tests: strip choice prefix

def test_strip_choice_prefix_removes_letter_prefix():
    """Verify common multiple-choice prefixes are removed."""
    assert strip_choice_prefix("A. Paris") == "Paris"
    assert strip_choice_prefix("b) Berlin") == "Berlin"
    assert strip_choice_prefix("C: Rome") == "Rome"
    assert strip_choice_prefix("d - Madrid") == "Madrid"

def test_strip_choice_prefix_does_not_touch_decimal_numbers():
    """Verify numeric decimals are not modified."""
    assert strip_choice_prefix("0.242") == "0.242"

# Unit tests: choices parsing

def test_parse_choices_accepts_list():
    """Verify list input is converted to stripped string list."""
    assert parse_choices([" A ", "B", 3]) == ["A", "B", "3"]

def test_parse_choices_parses_string_list_literal():
    """Verify string representation of list is parsed safely."""
    raw = "['Yes', 'No']"
    assert parse_choices(raw) == ["Yes", "No"]

def test_parse_choices_falls_back_to_manual_cleaning():
    """Verify fallback cleaning works if literal_eval fails."""
    raw = '["Yes" "No"]'
    out = parse_choices(raw)
    assert isinstance(out, list)
    assert len(out) >= 1

# Unit tests: file loading

def test_load_jsonl_raises_if_file_missing(tmp_path):
    """Verify missing file raises FileNotFoundError."""
    p = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        load_jsonl(str(p))

def test_load_jsonl_raises_if_empty(tmp_path):
    """Verify empty JSONL file raises ValueError."""
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        load_jsonl(str(p))

def test_load_jsonl_reads_valid_file(tmp_path):
    """Verify valid JSONL is loaded into DataFrame."""
    p = tmp_path / "ok.jsonl"
    _write_jsonl(p, [{"a": 1}, {"a": 2}])

    df = load_jsonl(str(p))

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df["a"].tolist() == [1, 2]

# Unit tests: row evaluation

def test_evaluate_row_index_mode_exact_match():
    """Verify correct answer is detected when model predicts exact choice text."""
    row = {
        "question": "Q",
        "choices": ["Paris", "Berlin", "Rome", "Madrid"],
        "answer": 0,
        "model_answer": "Paris",
    }

    is_correct, sample = evaluate_row(
        row,
        answer_column="answer",
        model_answer_column="model_answer",
        choices_column="choices",
        mode="index",
        ignore_case=True,
    )

    assert is_correct is True
    assert sample["correct"] == 1
    assert sample["expected"] == "paris"
    assert sample["predicted_norm"] == "paris"

def test_evaluate_row_index_mode_letter_only_answer():
    """Verify letter-only answers like 'A' are accepted."""
    row = {
        "question": "Q",
        "choices": ["Paris", "Berlin", "Rome", "Madrid"],
        "answer": 0,
        "model_answer": "A",
    }

    is_correct, sample = evaluate_row(
        row,
        answer_column="answer",
        model_answer_column="model_answer",
        choices_column="choices",
        mode="index",
        ignore_case=True,
    )

    assert is_correct is True
    assert sample["predicted_letter"] == "a"

def test_evaluate_row_index_mode_strips_prefix_like_A_dot():
    """Verify prefixes like 'A. Paris' are stripped and still match."""
    row = {
        "question": "Q",
        "choices": ["Paris", "Berlin", "Rome", "Madrid"],
        "answer": 0,
        "model_answer": "A. Paris",
    }

    is_correct, sample = evaluate_row(
        row,
        answer_column="answer",
        model_answer_column="model_answer",
        choices_column="choices",
        mode="index",
        ignore_case=True,
    )

    assert is_correct is True
    assert sample["predicted_norm"] == "paris"

def test_evaluate_row_text_mode_matches_string_answer():
    """Verify text mode compares literal correct answer string."""
    row = {
        "question": "Q",
        "choices": ["irrelevant"],
        "answer": "Newton",
        "model_answer": "newton",
    }

    is_correct, sample = evaluate_row(
        row,
        answer_column="answer",
        model_answer_column="model_answer",
        choices_column="choices",
        mode="text",
        ignore_case=True,
    )

    assert is_correct is True
    assert sample["expected"] == "newton"

def test_evaluate_row_case_sensitive_can_fail():
    """Verify ignore_case=False makes comparisons case-sensitive."""
    row = {
        "question": "Q",
        "choices": ["Paris"],
        "answer": 0,
        "model_answer": "paris",
    }

    is_correct, sample = evaluate_row(
        row,
        answer_column="answer",
        model_answer_column="model_answer",
        choices_column="choices",
        mode="index",
        ignore_case=False,
    )

    assert is_correct is False
    assert sample["correct"] == 0

# Unit tests: per-sample, mismatches and summary

def test_build_per_sample_accuracy_counts_correctness():
    """Verify per-sample accuracy builds deterministic totals and correct count."""
    df = pd.DataFrame(
        [
            {"question": "Q1", "choices": ["Yes", "No"], "answer": 0, "model_answer": "Yes"},
            {"question": "Q2", "choices": ["Yes", "No"], "answer": 1, "model_answer": "Yes"},
        ]
    )

    per_sample, total, correct = build_per_sample_accuracy(
        df,
        answer_column="answer",
        question_column="question",
        model_answer_column="model_answer",
        choices_column="choices",
        mode="index",
        ignore_case=True,
    )

    assert total == 2
    assert correct == 1
    assert len(per_sample) == 2
    assert per_sample[0]["accuracy.correct"] == 1
    assert per_sample[1]["accuracy.correct"] == 0

def test_collect_mismatches_limits_examples():
    """Verify collect_mismatches returns only first N incorrect samples."""
    per_sample = [
        {"accuracy.correct": 0},
        {"accuracy.correct": 0},
        {"accuracy.correct": 1},
    ]

    out = collect_mismatches(per_sample, max_mismatches=1)
    assert len(out) == 1

def test_collect_mismatches_raises_on_negative():
    """Verify negative max_mismatches raises ValueError."""
    with pytest.raises(ValueError):
        collect_mismatches([], max_mismatches=-1)

def test_summarize_results_returns_expected_fields():
    """Verify summary includes accuracy and counts."""
    metrics = summarize_results(total=4, correct=3)

    assert metrics["total"] == 4
    assert metrics["correct"] == 3
    assert metrics["incorrect"] == 1
    assert metrics["accuracy"] == 0.75

# Unit tests: save report

def test_save_report_jsonl_writes_file(tmp_path):
    """Verify report JSONL file is created and contains metrics lines."""
    out_path = tmp_path / "report.jsonl"
    save_report_jsonl({"accuracy": 1.0, "total": 2}, str(out_path))

    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "accuracy" in content
    assert "total" in content

# Offline test: evaluate_accuracy

def test_evaluate_accuracy_end_to_end(tmp_path):
    """Verify evaluate_accuracy produces correct metrics and per-sample output."""
    p = tmp_path / "results.jsonl"

    rows = [
        {"question": "Q1", "choices": ["Paris", "Berlin"], "answer": 0, "model_answer": "Paris"},
        {"question": "Q2", "choices": ["Paris", "Berlin"], "answer": 1, "model_answer": "Paris"},
    ]
    _write_jsonl(p, rows)

    out = evaluate_accuracy(
        input_path=str(p),
        answer_column="answer",
        question_column="question",
        model_answer_column="model_answer",
        choices_column="choices",
        mode="index",
        max_mismatches=10,
        output_path=None,
        ignore_case=True,
    )

    assert "metrics" in out
    assert "per_sample" in out
    assert "mismatches" in out

    assert out["metrics"]["total"] == 2
    assert out["metrics"]["correct"] == 1
    assert out["metrics"]["accuracy"] == 0.5
    assert len(out["per_sample"]) == 2