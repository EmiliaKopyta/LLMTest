from __future__ import annotations
import json
import pandas as pd
import pytest

from lib.evaluators.evaluate_bertscore import (
    normalize,
    load_jsonl_df,
    save_jsonl_df,
    validate_required_columns,
    extract_references_predictions,
    build_per_sample_bertscore,
    collect_bertscore_mismatches,
    summarize_bertscore,
    attach_bertscore_to_df,
    evaluate_bertscore,
)

# Helpers

def _write_jsonl(path, rows: list[dict]) -> None:
    """Write rows to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

class _FakeTensor:
    """Minimal tensor-like object that supports .item()."""
    def __init__(self, value: float):
        self._v = float(value)

    def item(self) -> float:
        return float(self._v)

class _FakeTensorList(list):
    """List with .mean().item() to mimic torch tensor behavior used in evaluator."""
    def mean(self):
        if not self:
            return _FakeTensor(0.0)
        return _FakeTensor(sum(x.item() for x in self) / len(self))

def _fake_scores(n: int):
    """
    Fake P/R/F1 with tensor-like API:
    - indexable elements have .item()
    - lists have .mean().item()
    """
    P = _FakeTensorList([_FakeTensor(0.9) for _ in range(n)])
    R = _FakeTensorList([_FakeTensor(0.8) for _ in range(n)])
    F1 = _FakeTensorList([_FakeTensor(0.85) for _ in range(n)])
    return P, R, F1

# Unit tests: normalize

def test_normalize_handles_none():
    """Verify normalize(None) returns empty string."""
    assert normalize(None) == ""

def test_normalize_strips_text():
    """Verify normalize strips whitespace from strings."""
    assert normalize("  hi  ") == "hi"

# Unit tests: load/save JSONL

def test_load_jsonl_df_raises_if_missing(tmp_path):
    """Verify missing file raises FileNotFoundError."""
    p = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        load_jsonl_df(str(p))

def test_load_jsonl_df_raises_if_empty(tmp_path):
    """Verify empty JSONL file raises ValueError."""
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        load_jsonl_df(str(p))

def test_load_jsonl_df_reads_valid(tmp_path):
    """Verify valid JSONL is loaded into DataFrame."""
    p = tmp_path / "ok.jsonl"
    _write_jsonl(p, [{"a": 1}, {"a": 2}])

    df = load_jsonl_df(str(p))

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df["a"].tolist() == [1, 2]

def test_save_jsonl_df_writes_file(tmp_path):
    """Verify DataFrame is saved to JSONL file."""
    out = tmp_path / "out.jsonl"
    df = pd.DataFrame([{"x": 1}, {"x": 2}])

    save_jsonl_df(df, str(out))

    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert '"x"' in content

# Unit tests: validation and extraction

def test_validate_required_columns_raises_when_missing():
    """Verify missing required columns raise KeyError."""
    df = pd.DataFrame([{"a": 1}])
    with pytest.raises(KeyError):
        validate_required_columns(df, ["a", "missing"])

def test_extract_references_predictions_returns_lists():
    """Verify reference/prediction lists are extracted and normalized."""
    df = pd.DataFrame(
        [
            {"ref": " hello ", "pred": "world"},
            {"ref": None, "pred": " test "},
        ]
    )

    refs, preds = extract_references_predictions(df, "ref", "pred")

    assert refs == ["hello", ""]
    assert preds == ["world", "test"]

# Unit tests: per-sample, mismatches and summary

def test_build_per_sample_bertscore_structure():
    """Verify per-sample BERTScore rows have correct keys and sample_id."""
    references = ["r1", "r2"]
    predictions = ["p1", "p2"]
    P, R, F1 = _fake_scores(2)

    rows = build_per_sample_bertscore(references, predictions, P, R, F1, prefix="bertscore")

    assert len(rows) == 2
    assert rows[0]["sample_id"] == 0
    assert "bertscore.precision" in rows[0]
    assert "bertscore.recall" in rows[0]
    assert "bertscore.f1" in rows[0]
    assert rows[0]["bertscore.reference"] == "r1"
    assert rows[0]["bertscore.predicted"] == "p1"

def test_collect_bertscore_mismatches_returns_lowest_f1_first():
    """Verify mismatches are sorted ascending by F1 and limited by max_mismatches."""
    per_sample = [
        {"sample_id": 0, "bertscore.f1": 0.9},
        {"sample_id": 1, "bertscore.f1": 0.1},
        {"sample_id": 2, "bertscore.f1": 0.5},
    ]

    mismatches = collect_bertscore_mismatches(per_sample, max_mismatches=2, f1_key="bertscore.f1")

    assert len(mismatches) == 2
    assert mismatches[0]["sample_id"] == 1
    assert mismatches[1]["sample_id"] == 2

def test_collect_bertscore_mismatches_raises_on_negative():
    """Verify negative max_mismatches raises ValueError."""
    with pytest.raises(ValueError):
        collect_bertscore_mismatches([], max_mismatches=-1)

def test_collect_bertscore_mismatches_raises_if_missing_key():
    """Verify missing f1_key raises KeyError."""
    with pytest.raises(KeyError):
        collect_bertscore_mismatches([{"sample_id": 0}], max_mismatches=1, f1_key="bertscore.f1")

def test_summarize_bertscore_returns_expected_fields():
    """Verify average metrics are computed and returned."""
    P, R, F1 = _fake_scores(3)
    metrics = summarize_bertscore(P, R, F1, total=3)

    assert metrics["total"] == 3
    assert metrics["bertscore_precision"] == 0.9
    assert metrics["bertscore_recall"] == 0.8
    assert metrics["bertscore_f1"] == 0.85

def test_attach_bertscore_to_df_adds_columns():
    """Verify BERTScore columns are attached to DataFrame."""
    df = pd.DataFrame([{"x": 1}, {"x": 2}])
    P, R, F1 = _fake_scores(2)

    out = attach_bertscore_to_df(df, P, R, F1)

    assert "bertscore_precision" in out.columns
    assert "bertscore_recall" in out.columns
    assert "bertscore_f1" in out.columns
    assert out["bertscore_f1"].tolist() == [0.85, 0.85]

# Offline test: evaluate_bertscore

def test_evaluate_bertscore_end_to_end_offline(monkeypatch, tmp_path):
    """
    Verify evaluate_bertscore runs offline:
    - loads jsonl
    - validates required columns
    - uses mocked BERTScore tensors
    - returns metrics/per_sample/mismatches
    """
    input_path = tmp_path / "input.jsonl"
    _write_jsonl(
        input_path,
        [
            {"summary": "ref1", "model_answer": "pred1"},
            {"summary": "ref2", "model_answer": "pred2"},
        ],
    )

    def fake_compute(references, predictions, lang="en", model_type="bert-base-uncased"):
        return _fake_scores(len(references))

    monkeypatch.setattr("lib.evaluators.evaluate_bertscore.compute_bertscore", fake_compute)

    out = evaluate_bertscore(
        input_path=str(input_path),
        reference_column="summary",
        model_answer_column="model_answer",
        max_mismatches=1,
        output_path=None,
    )

    assert "metrics" in out
    assert "per_sample" in out
    assert "mismatches" in out

    assert out["metrics"]["total"] == 2
    assert out["metrics"]["bertscore_f1"] == 0.85
    assert len(out["per_sample"]) == 2
    assert len(out["mismatches"]) == 1

def test_evaluate_bertscore_writes_output_when_output_path_given(monkeypatch, tmp_path):
    """Verify output JSONL is written when output_path is provided."""
    input_path = tmp_path / "input.jsonl"
    out_path = tmp_path / "scored.jsonl"

    _write_jsonl(
        input_path,
        [
            {"summary": "ref1", "model_answer": "pred1"},
            {"summary": "ref2", "model_answer": "pred2"},
        ],
    )

    def fake_compute(references, predictions, lang="en", model_type="bert-base-uncased"):
        return _fake_scores(len(references))

    monkeypatch.setattr("lib.evaluators.evaluate_bertscore.compute_bertscore", fake_compute)

    out = evaluate_bertscore(
        input_path=str(input_path),
        reference_column="summary",
        model_answer_column="model_answer",
        output_path=str(out_path),
    )

    assert out_path.exists()

    df_scored = pd.read_json(out_path, lines=True)
    assert "bertscore_f1" in df_scored.columns
    assert df_scored["bertscore_f1"].tolist() == [0.85, 0.85]