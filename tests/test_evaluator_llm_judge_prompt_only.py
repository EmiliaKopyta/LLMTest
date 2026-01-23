from __future__ import annotations
import json
from pathlib import Path
import pytest

from lib.evaluators.evaluate_llm_judge_prompt_only import (
    _is_float,
    build_judge_prompt,
    collect_mismatches,
    evaluate_llm_judge_prompt_only,
    save_jsonl,
)

# Helpers

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _read_jsonl(path: Path) -> list[dict]:
    """Read JSONL into list[dict]."""
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

# Unit tests: class helpers

def test_is_float_accepts_dot_and_comma():
    """Verify _is_float detects numeric strings, including comma decimals."""
    assert _is_float("0.25") is True
    assert _is_float("1,25") is True
    assert _is_float("  2  ") is True
    assert _is_float("-3.14") is True

def test_is_float_rejects_non_numeric_text():
    """Verify _is_float rejects non-numeric strings."""
    assert _is_float("abc") is False
    assert _is_float("A") is False
    assert _is_float("") is False

def test_build_judge_prompt_contains_sections():
    """Verify judge prompt includes expected sections and final instruction."""
    judge_prompt = "Score from 0 to 1."
    prompt = "What is 2+2?"
    model_answer = "4"

    out = build_judge_prompt(judge_prompt, prompt, model_answer)

    assert "Score from 0 to 1." in out
    assert "Prompt:" in out
    assert "Model Answer:" in out
    assert "Please respond according to the criteria above." in out
    assert "What is 2+2?" in out
    assert "\n4\n" in out

# Unit tests: mismatches

def test_collect_mismatches_sorts_numeric_judgments():
    """Verify that numeric judgments are sorted ascending and sliced to max_mismatches."""
    judged = [
        {"judge.judgment": "0.9"},
        {"judge.judgment": "0.1"},
        {"judge.judgment": "0.5"},
    ]

    out = collect_mismatches(judged, max_mismatches=2)

    assert len(out) == 2
    assert out[0]["judge.judgment"] == "0.1"
    assert out[1]["judge.judgment"] == "0.5"

def test_collect_mismatches_falls_back_for_text_judgments():
    """Verify that non-numeric judgments are sorted in order (first N)."""
    judged = [
        {"judge.judgment": "bad"},
        {"judge.judgment": "ok"},
        {"judge.judgment": "great"},
    ]

    out = collect_mismatches(judged, max_mismatches=2)

    assert len(out) == 2
    assert out[0]["judge.judgment"] == "bad"
    assert out[1]["judge.judgment"] == "ok"

def test_collect_mismatches_raises_for_negative_limit():
    """Verify negative max_mismatches raises ValueError."""
    with pytest.raises(ValueError):
        collect_mismatches([], max_mismatches=-1)

# Unit tests: save a jsonl file

def test_save_jsonl_writes_one_object_per_line(tmp_path: Path):
    """Verify save_jsonl writes a correct JSONL file (one JSON per line)."""
    rows = [
        {"a": 1},
        {"a": 2},
        {"a": 3},
    ]
    out_path = tmp_path / "out.jsonl"

    save_jsonl(rows, str(out_path))

    assert out_path.exists()
    loaded = _read_jsonl(out_path)
    assert loaded == rows

def test_save_jsonl_creates_folder(tmp_path: Path):
    """Verify save_jsonl creates missing parent directory."""
    rows = [{"x": 1}]
    nested = tmp_path / "nested" / "results.jsonl"

    save_jsonl(rows, str(nested))

    assert nested.exists()
    assert _read_jsonl(nested) == rows

# Offline test: _evaluate_llm_judge_prompt_only

@pytest.mark.asyncio
async def test_evaluate_llm_judge_prompt_only_offline(monkeypatch, tmp_path: Path):
    """
    Verify evaluate_llm_judge_prompt_only works fully offline:
    - reads JSONL
    - builds judge prompts
    - uses mocked run_llm_judgment
    - returns metrics/per_sample/mismatches
    - optionally writes output JSONL
    """
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "judged.jsonl"

    _write_jsonl(
        input_path,
        [
            {"prompt": "P1", "model_answer": "A1"},
            {"prompt": "P2", "model_answer": "A2"},
            {"prompt": "P3", "model_answer": "A3"},
        ],
    )

    async def fake_run_llm_judgment(full_prompt: str, prompt_handler):
        if "P1" in full_prompt:
            return "0.9"
        if "P2" in full_prompt:
            return "0.1"
        return "0.5"

    monkeypatch.setattr("lib.evaluators.evaluate_llm_judge_prompt_only.run_llm_judgment", fake_run_llm_judgment)

    class FakePromptHandler:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("lib.evaluators.evaluate_llm_judge_prompt_only.PromptHandler", FakePromptHandler)

    out = await evaluate_llm_judge_prompt_only(
        input_path=str(input_path),
        judge_prompt="Judge it.",
        model_provider="openai",
        model_name="gpt-4o",
        prompt_column="prompt",
        max_mismatches=2,
        output_path=str(output_path),
    )

    assert "metrics" in out
    assert "per_sample" in out
    assert "mismatches" in out

    assert out["metrics"]["samples"] == 3
    assert len(out["per_sample"]) == 3

    first = out["per_sample"][0]
    assert "sample_id" in first
    assert "judge.prompt" in first
    assert "judge.model_answer" in first
    assert "judge.judgment" in first

    assert len(out["mismatches"]) == 2
    assert out["mismatches"][0]["judge.judgment"] == "0.1"
    assert out["mismatches"][1]["judge.judgment"] == "0.5"

    assert output_path.exists()
    saved = _read_jsonl(output_path)
    assert len(saved) == 3
    assert saved[0]["judge.prompt"] == "P1"