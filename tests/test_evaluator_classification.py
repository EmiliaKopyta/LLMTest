from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import pytest

from lib.evaluators.evaluate_classification import (
    load_jsonl,
    compute_stats,
    build_per_sample,
    collect_mismatches,
    build_confusion_matrix,
    plot_confusion_matrix,
    evaluate_classification,
)

# Helpers

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write records to JSONL for evaluator tests."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _toy_df() -> pd.DataFrame:
    """A small deterministic classification dataframe:
    labels: cat/dog
    predictions: 3 correct, 1 wrong
    """
    return pd.DataFrame(
        [
            {"text": "t1", "label": "cat", "model_answer": "cat"},
            {"text": "t2", "label": "cat", "model_answer": "dog"},
            {"text": "t3", "label": "dog", "model_answer": "dog"},
            {"text": "t4", "label": "dog", "model_answer": "dog"},
        ]
    )

# Unit tests: load the file

def test_load_jsonl_raises_if_missing_file(tmp_path: Path):
    """Verify load_jsonl raises FileNotFoundError if file is missing."""
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        load_jsonl(str(missing))

def test_load_jsonl_raises_if_empty_file(tmp_path: Path):
    """Verify load_jsonl raises ValueError if JSONL is empty."""
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_jsonl(str(p))

def test_load_jsonl_loads_valid_jsonl(tmp_path: Path):
    """Verify load_jsonl loads JSONL into a non-empty DataFrame."""
    p = tmp_path / "data.jsonl"
    _write_jsonl(p, [{"a": 1}, {"a": 2}])

    df = load_jsonl(str(p))

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df["a"].tolist() == [1, 2]

# Unit tests: compute stats

def test_compute_stats_raises_if_missing_columns():
    """Verify compute_stats raises KeyError if required columns are missing."""
    df = pd.DataFrame([{"x": 1}])
    with pytest.raises(KeyError):
        compute_stats(df, label_column="label", model_answer_column="model_answer")

def test_compute_stats_raises_if_empty_df():
    """Verify compute_stats raises ValueError if DataFrame is empty."""
    df = pd.DataFrame(columns=["label", "model_answer"])
    with pytest.raises(ValueError):
        compute_stats(df)

def test_compute_stats_returns_expected_metrics():
    """Verify compute_stats returns correct macro metrics and per-class stats."""
    df = _toy_df()
    out = compute_stats(df, label_column="label", model_answer_column="model_answer")

    assert out["total"] == 4
    assert out["correct"] == 3
    assert out["incorrect"] == 1
    assert out["accuracy"] == 0.75

    assert "cat" in out["per_class"]
    assert "dog" in out["per_class"]

    # cat: tp=1 (t1), fn=1 (t2 wrong), fp=0 (never predicted cat wrongly)
    cat = out["per_class"]["cat"]
    assert cat["tp"] == 1
    assert cat["fn"] == 1
    assert cat["fp"] == 0

    # dog: tp=2 (t3,t4), fp=1 (t2 predicted dog but should be cat), fn=0
    dog = out["per_class"]["dog"]
    assert dog["tp"] == 2
    assert dog["fp"] == 1
    assert dog["fn"] == 0

# Unit tests: per-sample and mismatches

def test_build_per_sample_returns_expected_rows():
    """Verify build_per_sample produces rows that can be merged with sample_id and correctness."""
    df = _toy_df()
    per = build_per_sample(df, label_column="label", model_answer_column="model_answer")

    assert isinstance(per, list)
    assert len(per) == 4
    assert per[0]["sample_id"] == 0
    assert per[0]["classification.correct"] == 1
    assert per[1]["classification.correct"] == 0
    assert per[0]["classification.expected"] == "cat"
    assert per[0]["classification.predicted"] == "cat"

def test_collect_mismatches_respects_limit():
    """Verify collect_mismatches returns up to max_mismatches incorrect rows."""
    df = _toy_df()
    mism = collect_mismatches(df, max_mismatches=1)

    assert isinstance(mism, list)
    assert len(mism) == 1
    assert mism[0]["expected"] == "cat"
    assert str(mism[0]["predicted"]).strip().lower() == "dog"

def test_collect_mismatches_raises_on_negative_limit():
    """Verify collect_mismatches raises ValueError for negative max_mismatches."""
    df = _toy_df()
    with pytest.raises(ValueError):
        collect_mismatches(df, max_mismatches=-1)

# Unit tests: confusion matrix

def test_build_confusion_matrix_counts_correctly():
    """Verify confusion matrix counts true->predicted occurrences."""
    df = _toy_df()
    cm = build_confusion_matrix(df, label_column="label", model_answer_column="model_answer")

    # true cat -> predicted cat = 1, predicted dog = 1
    assert cm["cat"]["cat"] == 1
    assert cm["cat"]["dog"] == 1

    # true dog -> predicted dog = 2
    assert cm["dog"]["dog"] == 2

def test_plot_confusion_matrix_raises_on_empty():
    """Verify plot_confusion_matrix raises ValueError when dict is empty."""
    with pytest.raises(ValueError):
        plot_confusion_matrix({})

# Offline test: evaluate_classification

def test_evaluate_classification_end_to_end_offline(tmp_path: Path):
    """Verify evaluate_classification produces expected structure and mismatch count."""
    p = tmp_path / "input.jsonl"
    _write_jsonl(
        p,
        [
            {"text": "t1", "label": "cat", "model_answer": "cat"},
            {"text": "t2", "label": "cat", "model_answer": "dog"},
            {"text": "t3", "label": "dog", "model_answer": "dog"},
        ],
    )

    out = evaluate_classification(
        input_path=str(p),
        label_column="label",
        model_answer_column="model_answer",
        include_confusion=False,
        max_mismatches=10,
        output_path=None,
    )

    assert "metrics" in out
    assert "per_sample" in out
    assert "mismatches" in out

    assert out["metrics"]["total"] == 3
    assert out["metrics"]["correct"] == 2
    assert out["metrics"]["accuracy"] == round(2 / 3, 4)

    assert len(out["per_sample"]) == 3
    assert sum(r["classification.correct"] for r in out["per_sample"]) == 2

    assert len(out["mismatches"]) == 1

def test_evaluate_classification_confusion_matrix_path_offline(monkeypatch, tmp_path: Path):
    """
    Verify include_confusion=True attaches confusion matrix and does not render plots in tests.
    Patched plot_confusion_matrix to avoid plt.show().
    """
    p = tmp_path / "input.jsonl"
    _write_jsonl(
        p,
        [
            {"text": "t1", "label": "cat", "model_answer": "cat"},
            {"text": "t2", "label": "cat", "model_answer": "dog"},
        ],
    )

    calls = {"called": False}

    def fake_plot(_conf):
        calls["called"] = True

    monkeypatch.setattr("lib.evaluators.evaluate_classification.plot_confusion_matrix", fake_plot)

    out = evaluate_classification(
        input_path=str(p),
        label_column="label",
        model_answer_column="model_answer",
        include_confusion=True,
        max_mismatches=10,
        output_path=None,
    )

    assert "confusion_matrix" in out["metrics"]
    assert calls["called"] is True

def test_evaluate_classification_writes_output_when_output_path_given(tmp_path: Path):
    """Verify output_path causes writing per_sample JSONL rows."""
    p = tmp_path / "input.jsonl"
    out_path = tmp_path / "per_sample.jsonl"

    _write_jsonl(
        p,
        [
            {"text": "x", "label": "a", "model_answer": "a"},
            {"text": "y", "label": "b", "model_answer": "a"},
        ],
    )

    out = evaluate_classification(
        input_path=str(p),
        label_column="label",
        model_answer_column="model_answer",
        include_confusion=False,
        output_path=str(out_path),
    )

    assert out_path.exists()

    df_out = pd.read_json(out_path, lines=True)
    assert len(df_out) == len(out["per_sample"])