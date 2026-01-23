from __future__ import annotations

import os
import asyncio
import pandas as pd
import pytest

from lib.TestRunner import TestRunner, validate_prompt_output

# Fake Prompt Handler and fixtures

class FakePromptHandler:
    """Fake PromptHandler used to avoid ModelRegistry and real Agent setup. Necessary because TestRunner creates PromptHandler in __init__()."""
    def __init__(self, *args, **kwargs):
        pass

    async def generate_response(self, prompt):
        return None

@pytest.fixture(autouse=True)
def _patch_prompt_handler(monkeypatch):
    """Patch PromptHandler used inside TestRunner to prevent external dependencies"""
    monkeypatch.setattr("lib.TestRunner.PromptHandler", FakePromptHandler)

@pytest.fixture
def runner(tmp_path) -> TestRunner:
    """A TestRunner instance configured to write results into a temporary folder."""
    out_file = tmp_path / "out.jsonl"

    def _dummy_prompt_builder(row) -> str:
        return row["prompt"]

    return TestRunner(
        test_name="t",
        test_dataset_path="dummy",
        model_provider="openai",
        model_name="m",
        prompt_builder=_dummy_prompt_builder,
        evaluation_output_path=str(out_file),
        output_format="jsonl",
        max_concurrency=1,
    )

def _dummy_df(n: int = 5) -> pd.DataFrame:
    """A small deterministic DataFrame for selection tests."""
    return pd.DataFrame({"prompt": [f"p{i}" for i in range(1, n + 1)]})

# Unit tests: prompt output validation

def test_validate_prompt_output_accepts_none():
    """Verify that None prompt is accepted (treated as skip)."""
    validate_prompt_output(None)

def test_validate_prompt_output_accepts_str():
    """Verify that a plain string prompt is accepted."""
    validate_prompt_output("hello")

def test_validate_prompt_output_accepts_list_tuple_dict():
    """Verify that structured prompts are accepted (list/tuple/dict)."""
    validate_prompt_output(["hello"])
    validate_prompt_output(("hello",))
    validate_prompt_output({"type": "user", "content": "hello"})

def test_validate_prompt_output_raises_on_invalid_type():
    """Verify that unsupported prompt types raise TypeError."""
    with pytest.raises(TypeError):
        validate_prompt_output(123)

# Unit tests: row selection

def test_select_rows_single_index(runner):
    """Verify selection with a single index like '1' returns exactly one row."""
    df = _dummy_df(5)
    out = runner._select_rows(df, "1")

    assert len(out) == 1
    assert out.iloc[0]["prompt"] == "p1"

def test_select_rows_range_from_start(runner):
    """Verify selection 'start-3' returns the first 3 rows."""
    df = _dummy_df(5)
    out = runner._select_rows(df, "start-3")

    assert len(out) == 3
    assert out["prompt"].tolist() == ["p1", "p2", "p3"]

def test_select_rows_range_to_end(runner):
    """Verify selection '2-end' returns rows from index 2 to the end (1-based)."""
    df = _dummy_df(5)
    out = runner._select_rows(df, "2-end")

    assert out["prompt"].tolist() == ["p2", "p3", "p4", "p5"]

def test_select_rows_random_n(monkeypatch, runner):
    """Verify selection 'random:N' returns N rows (deterministic via patch)."""
    df = _dummy_df(5)

    # sampling deterministic for the test
    monkeypatch.setattr(pd.DataFrame, "sample", lambda self, n: self.iloc[:n])

    out = runner._select_rows(df, "random:2")

    assert len(out) == 2
    assert out["prompt"].tolist() == ["p1", "p2"]

def test_select_rows_invalid_format_raises(runner):
    """Verify invalid selection format raises ValueError."""
    df = _dummy_df(5)

    with pytest.raises(ValueError):
        runner._select_rows(df, "bad-format")

# Unit tests: save results

def test_save_results_jsonl_to_custom_path(tmp_path):
    """Verify JSONL results are written when output_format is jsonl."""
    out_file = tmp_path / "out.jsonl"

    def _builder(row) -> str:
        return row["prompt"]

    runner = TestRunner(
        test_name="t",
        test_dataset_path="dummy",
        model_provider="openai",
        model_name="m",
        prompt_builder=_builder,
        evaluation_output_path=str(out_file),
        output_format="jsonl",
        max_concurrency=1,
    )

    df = pd.DataFrame({"a": [1, 2]})
    path = runner.save_results(df)

    assert os.path.exists(path)
    assert path.endswith(".jsonl")

def test_save_results_csv_to_custom_path(tmp_path):
    """Verify CSV results are written when output_format is csv."""
    out_file = tmp_path / "out.csv"

    def _builder(row) -> str:
        return row["prompt"]

    runner = TestRunner(
        test_name="t",
        test_dataset_path="dummy",
        model_provider="openai",
        model_name="m",
        prompt_builder=_builder,
        evaluation_output_path=str(out_file),
        output_format="csv",
        max_concurrency=1,
    )

    df = pd.DataFrame({"a": [1, 2]})
    path = runner.save_results(df)

    assert os.path.exists(path)
    assert path.endswith(".csv")

def test_save_results_unsupported_format_raises(tmp_path):
    """Verify unsupported output formats raise ValueError."""
    out_file = tmp_path / "out.xyz"

    def _builder(row) -> str:
        return row["prompt"]

    runner = TestRunner(
        test_name="t",
        test_dataset_path="dummy",
        model_provider="openai",
        model_name="m",
        prompt_builder=_builder,
        evaluation_output_path=str(out_file),
        output_format="xyz",
        max_concurrency=1,
    )

    df = pd.DataFrame({"a": [1]})

    with pytest.raises(ValueError):
        runner.save_results(df)

# Offline test: run_model

def test_run_model_pipeline_offline(monkeypatch, tmp_path):
    """
    Verify run_model() executes the pipeline and saves a file without calling real LLMs.

    This test patches:
    - dataset loading (returns a deterministic DataFrame)
    - answer generation (returns fixed answers)
    """
    out_file = tmp_path / "results.jsonl"

    def _builder(row) -> str:
        return row["prompt"]

    runner = TestRunner(
        test_name="offline",
        test_dataset_path="dummy",
        model_provider="openai",
        model_name="m",
        prompt_builder=_builder,
        evaluation_output_path=str(out_file),
        output_format="jsonl",
        max_concurrency=1,
    )

    monkeypatch.setattr(runner, "load_dataset", lambda selection=None: setattr(runner, "df", _dummy_df(3)))

    async def _fake_generate_answers(df, prompt_builder, handler, max_concurrency):
        return ["a1", "a2", "a3"]

    monkeypatch.setattr("lib.TestRunner.generate_answers", _fake_generate_answers)

    saved_path = asyncio.run(runner.run_model())
    assert os.path.exists(saved_path)

    df_saved = pd.read_json(saved_path, lines=True)
    assert "model_answer" in df_saved.columns
    assert df_saved["model_answer"].tolist() == ["a1", "a2", "a3"]