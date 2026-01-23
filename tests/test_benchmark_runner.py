from __future__ import annotations
import os
import json
import asyncio
import pandas as pd
import pytest

from lib.BenchmarkRunner import (
    ModelSpec,
    EvaluationSpec,
    BenchmarkRunner,
    load_results_df,
    attach_per_sample,
    run_evaluation,
    resolve_benchmark_path,
    save_benchmark_dataset,
    save_benchmark_report,
    _ext_from_format,
)

# Helpers

def _write_jsonl(path, rows: list[dict]) -> None:
    """Write a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _dummy_results_rows() -> list[dict]:
    """A tiny set of model outputs."""
    return [
        {"question": "Q1", "model_answer": "A1"},
        {"question": "Q2", "model_answer": "A2"},
    ]

# Unit tests: loading results

def test_load_results_df_reads_jsonl_and_adds_metadata(tmp_path):
    """Verify JSONL loading works and benchmark metadata columns are attached."""
    results_path = tmp_path / "results.jsonl"
    _write_jsonl(results_path, _dummy_results_rows())

    model = ModelSpec(provider="openai", model_name="gpt-test")
    df = load_results_df(str(results_path), test_name="bench1", model=model, output_format="jsonl")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2

    assert "benchmark" in df.columns
    assert "model_provider" in df.columns
    assert "model_name" in df.columns
    assert "model" in df.columns
    assert "sample_id" in df.columns

    assert df.iloc[0]["benchmark"] == "bench1"
    assert df.iloc[0]["model_provider"] == "openai"
    assert df.iloc[0]["model_name"] == "gpt-test"
    assert df.iloc[0]["model"] == "openai:gpt-test"
    assert df["sample_id"].tolist() == [0, 1]

def test_load_results_df_reads_csv(tmp_path):
    """Verify CSV loading works."""
    results_path = tmp_path / "results.csv"
    pd.DataFrame(_dummy_results_rows()).to_csv(results_path, index=False)

    model = ModelSpec(provider="openai", model_name="gpt-test")
    df = load_results_df(str(results_path), test_name="bench1", model=model, output_format="csv")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "benchmark" in df.columns
    assert df.iloc[1]["model_answer"] == "A2"

# Unit tests: attachment of per-sample metrics

def test_attach_per_sample_merges_on_sample_id():
    """Verify per-sample metrics merge correctly into the results dataset."""
    df = pd.DataFrame(
        {
            "sample_id": [0, 1],
            "model_answer": ["A1", "A2"],
        }
    )

    per_sample = [
        {"sample_id": 0, "score": 1},
        {"sample_id": 1, "score": 0},
    ]

    out = attach_per_sample(df, per_sample)

    assert "score" in out.columns
    assert out["score"].tolist() == [1, 0]

def test_attach_per_sample_raises_if_missing_sample_id():
    """Verify attach_per_sample raises ValueError if sample_id is missing."""
    df = pd.DataFrame({"sample_id": [0], "model_answer": ["A1"]})
    per_sample = [{"id": 0, "score": 1}]

    with pytest.raises(ValueError):
        attach_per_sample(df, per_sample)

# Unit tests: file extensions

def test_ext_from_format_accepts_supported_formats():
    """Verify supported output formats are accepted."""
    assert _ext_from_format("jsonl") == "jsonl"
    assert _ext_from_format("json") == "json"
    assert _ext_from_format("csv") == "csv"

def test_ext_from_format_raises_on_invalid_format():
    """Verify invalid output formats raise ValueError."""
    with pytest.raises(ValueError):
        _ext_from_format("xml")

# Unit tests: benchmark path

def test_resolve_benchmark_path_returns_custom_output_path(tmp_path):
    """Verify resolve_benchmark_path returns output_path directly when provided."""
    custom = str(tmp_path / "custom.jsonl")
    out = resolve_benchmark_path(
        benchmark_name="bench",
        output_format="jsonl",
        kind="dataset",
        base_dir=str(tmp_path),
        timestamp="TS",
        output_path=custom,
    )
    assert out == custom

def test_resolve_benchmark_path_creates_folder_and_builds_path(tmp_path):
    """Verify resolve_benchmark_path creates folder and builds default file path."""
    out = resolve_benchmark_path(
        benchmark_name="bench",
        output_format="jsonl",
        kind="dataset",
        base_dir=str(tmp_path),
        timestamp="TS",
        output_path=None,
    )

    assert out.endswith(os.path.join("bench", "dataset_TS.jsonl"))
    assert os.path.isdir(os.path.join(str(tmp_path), "bench"))

# Unit tests: save benchmark dataset and report

def test_save_benchmark_dataset_writes_jsonl(tmp_path):
    """Verify benchmark dataset is saved as JSONL."""
    df = pd.DataFrame({"a": [1, 2]})
    out_path = tmp_path / "dataset.jsonl"

    saved = save_benchmark_dataset(df, str(out_path), "jsonl")

    assert os.path.exists(saved)
    assert saved.endswith(".jsonl")

def test_save_benchmark_report_writes_json(tmp_path):
    """Verify benchmark report is saved as JSON."""
    report = [{"model": "openai:m1", "metrics": {"acc": 1.0}}]
    out_path = tmp_path / "report.json"

    saved = save_benchmark_report(report, str(out_path), "json")

    assert os.path.exists(saved)
    with open(saved, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded[0]["model"] == "openai:m1"

def test_save_benchmark_dataset_raises_on_invalid_format(tmp_path):
    """Verify invalid dataset output formats raise ValueError."""
    df = pd.DataFrame({"a": [1]})
    out_path = tmp_path / "dataset.xyz"

    with pytest.raises(ValueError):
        save_benchmark_dataset(df, str(out_path), "xyz")

def test_save_benchmark_report_raises_on_invalid_format(tmp_path):
    """Verify invalid report output formats raise ValueError."""
    out_path = tmp_path / "report.xyz"

    with pytest.raises(ValueError):
        save_benchmark_report([{"x": 1}], str(out_path), "xyz")

# Unit tests: evaluation for sync and async evaluators

def test_run_evaluation_supports_sync_evaluator():
    """Verify run_evaluation works with a synchronous evaluator function."""
    def evaluator(_path: str):
        return {
            "metrics": {"acc": 0.5},
            "per_sample": [{"sample_id": 0, "score": 1}],
            "mismatches": ["m1"],
        }

    eval_spec = EvaluationSpec(name="sync_eval", evaluator=evaluator)
    model = ModelSpec(provider="openai", model_name="m1")

    report_row, per_sample = asyncio.run(
        run_evaluation(eval_spec, "dummy.jsonl", "bench", model, system_prompt="")
    )

    assert report_row["evaluation"] == "sync_eval"
    assert report_row["metrics"]["acc"] == 0.5
    assert report_row["mismatches"] == ["m1"]
    assert per_sample[0]["sample_id"] == 0

def test_run_evaluation_supports_async_evaluator():
    """Verify run_evaluation works with an async evaluator function."""
    async def evaluator(_path: str):
        return {
            "metrics": {"acc": 1.0},
            "per_sample": [{"sample_id": 0, "score": 1}],
        }

    eval_spec = EvaluationSpec(name="async_eval", evaluator=evaluator)
    model = ModelSpec(provider="openai", model_name="m1")

    report_row, per_sample = asyncio.run(
        run_evaluation(eval_spec, "dummy.jsonl", "bench", model, system_prompt="")
    )

    assert report_row["evaluation"] == "async_eval"
    assert report_row["metrics"]["acc"] == 1.0
    assert report_row["mismatches"] == []
    assert isinstance(per_sample, list)

# Offline test: run

def test_benchmark_runner_run_offline(monkeypatch, tmp_path):
    """
    Verify BenchmarkRunner.run pipeline offline:
    - uses TestRunner.run_model() (patched)
    - loads model results into df
    - runs evaluation and attaches per-sample columns
    - saves dataset and report
    """
    results_path = tmp_path / "results.jsonl"
    _write_jsonl(results_path, _dummy_results_rows())

    class FakeTestRunner:
        def __init__(self, *args, **kwargs):
            pass

        async def run_model(self, selection=None):
            return str(results_path)

    monkeypatch.setattr("lib.BenchmarkRunner.TestRunner", FakeTestRunner)

    def evaluator(_path: str):
        return {
            "metrics": {"acc": 0.5},
            "per_sample": [
                {"sample_id": 0, "score": 1},
                {"sample_id": 1, "score": 0},
            ],
        }

    models = [ModelSpec(provider="openai", model_name="m1")]
    evals = [EvaluationSpec(name="acc_eval", evaluator=evaluator)]

    runner = BenchmarkRunner(
        test_name="bench1",
        dataset_path="dummy",
        prompt_builder=lambda row: "x",
        models=models,
        evaluations=evals,
        system_prompt="",
        output_dir=str(tmp_path),
        output_format="jsonl",
        report_format="json",
        max_concurrency=1,
    )

    summary = asyncio.run(runner.run(selection="1"))

    assert summary["benchmark"] == "bench1"
    assert os.path.exists(summary["dataset_path"])
    assert os.path.exists(summary["report_path"])

    df_out = pd.read_json(summary["dataset_path"], lines=True)
    assert "score" in df_out.columns
    assert df_out["score"].tolist() == [1, 0]

    with open(summary["report_path"], "r", encoding="utf-8") as f:
        report = json.load(f)

    assert isinstance(report, list)
    assert len(report) == 1
    assert report[0]["evaluation"] == "acc_eval"
    assert report[0]["metrics"]["acc"] == 0.5