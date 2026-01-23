from __future__ import annotations
import json
from pathlib import Path
import pytest

from lib.evaluators.evaluate_llm_judge_scale_01 import (
    parse_score,
    summarize_results,
    collect_invalid_cases,
    collect_judge_mismatches,
    build_per_sample_judge,
    save_jsonl,
    evaluate_llm_judge_scale_01,
)

# Helpers

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows to JSONL (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _read_jsonl(path: Path) -> list[dict]:
    """Read JSONL into list of dicts."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

# Unit tests: score parsing

@pytest.mark.parametrize(
    "raw, expected_score, expected_invalid",
    [
        ("0.5", 0.5, False),
        ("0,7", 0.7, False),
        (" Score: 0.4/1 ", 0.4, False),
        ("Quality=1.0 (perfect)", 1.0, False),
        ("no number here", 0.0, True),
        ("", 0.0, True),
        (None, 0.0, True),
    ],
)
def test_parse_score_parses_numeric_and_fallbacks(raw, expected_score, expected_invalid):
    """Verify parse_score extracts floats, supports commas, and flags invalid text."""
    score, invalid = parse_score(raw)
    assert score == expected_score
    assert invalid == expected_invalid

# Unit tests: summarize and collectors

def test_summarize_results_computes_average_and_invalid_count():
    """Verify summarize_results returns correct aggregate metrics."""
    judged = [
        {"score": 0.2, "invalid": False},
        {"score": 0.8, "invalid": False},
        {"score": 0.0, "invalid": True},
    ]

    metrics = summarize_results(judged)

    assert metrics["samples"] == 3
    assert metrics["average_score"] == round((0.2 + 0.8 + 0.0) / 3, 4)
    assert metrics["invalid_count"] == 1

def test_collect_invalid_cases_filters_only_invalid():
    """Verify collect_invalid_cases returns only entries marked invalid."""
    judged = [
        {"score": 0.5, "invalid": False},
        {"score": 0.0, "invalid": True},
        {"score": 0.7, "invalid": False},
        {"score": 0.1, "invalid": True},
    ]

    out = collect_invalid_cases(judged)
    assert len(out) == 2
    assert all(x["invalid"] for x in out)

def test_collect_judge_mismatches_sorts_by_score_and_slices():
    """Verify mismatches are the lowest-scoring samples."""
    judged = [
        {"sample_id": 0, "score": 0.9},
        {"sample_id": 1, "score": 0.1},
        {"sample_id": 2, "score": 0.5},
    ]

    out = collect_judge_mismatches(judged, max_mismatches=2)

    assert len(out) == 2
    assert out[0]["score"] == 0.1
    assert out[1]["score"] == 0.5

def test_collect_judge_mismatches_raises_for_negative_limit():
    """Verify negative mismatch limit raises ValueError."""
    judged = [{"score": 0.1}]
    with pytest.raises(ValueError):
        collect_judge_mismatches(judged, max_mismatches=-1)

# Unit tests: build_per_sample_judge

def test_build_per_sample_judge_outputs_merge_friendly_schema():
    """Verify build_per_sample_judge returns rows keyed by sample_id with judge fields."""
    judged = [
        {
            "sample_id": 0,
            "score": 0.3,
            "invalid": False,
            "prompt": "P1",
            "model_answer": "A1",
            "reference": "R1",
        },
        {
            "sample_id": 1,
            "score": 0.8,
            "invalid": True,
            "prompt": "P2",
            "model_answer": "A2",
            "reference": "R2",
        },
    ]

    rows = build_per_sample_judge(judged, prefix="judge")

    assert len(rows) == 2
    assert rows[0]["sample_id"] == 0
    assert rows[0]["judge.score"] == 0.3
    assert rows[0]["judge.invalid"] == 0
    assert rows[0]["judge.prompt"] == "P1"
    assert rows[0]["judge.model_answer"] == "A1"
    assert rows[0]["judge.reference"] == "R1"

    assert rows[1]["judge.invalid"] == 1

# Unit tests: save to jsonl

def test_save_jsonl_writes_one_object_per_line(tmp_path: Path):
    """Verify save_jsonl writes valid JSONL format (one dict per line)."""
    out_path = tmp_path / "out" / "judged.jsonl"

    rows = [
        {"a": 1},
        {"b": 2},
        {"c": 3},
    ]

    save_jsonl(rows, str(out_path))

    assert out_path.exists()
    loaded = _read_jsonl(out_path)
    assert loaded == rows

# Offline test: evaluate_llm_judge_scale_01

@pytest.mark.asyncio
async def test_evaluate_llm_judge_scale_01_offline(monkeypatch, tmp_path: Path):
    """
    Verify evaluate_llm_judge_scale_01 runs fully offline:
    - loads JSONL
    - creates PromptHandler (patched)
    - uses judge_example (patched)
    - returns metrics/per_sample/mismatches/invalid_cases
    - writes output if output_path provided
    """
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "out" / "judged_results.jsonl"

    _write_jsonl(
        input_path,
        [
            {"prompt": "P1", "model_answer": "A1", "reference": "R1"},
            {"prompt": "P2", "model_answer": "A2", "reference": "R2"},
            {"prompt": "P3", "model_answer": "A3", "reference": "R3"},
        ],
    )

    class FakePromptHandler:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("lib.evaluators.evaluate_llm_judge_scale_01.PromptHandler", FakePromptHandler)

    async def fake_judge_example(example, judge_prompt, prompt_handler, sample_id, **kwargs):
        mapping = {
            0: (0.9, False),
            1: (0.1, False),
            2: (0.0, True),
        }
        score, invalid = mapping[int(sample_id)]
        return {
            "sample_id": int(sample_id),
            "prompt": example.get("prompt", ""),
            "model_answer": example.get("model_answer", ""),
            "reference": example.get("reference", ""),
            "score": float(score),
            "invalid": bool(invalid),
        }

    monkeypatch.setattr("lib.evaluators.evaluate_llm_judge_scale_01.judge_example", fake_judge_example)

    out = await evaluate_llm_judge_scale_01(
        input_path=str(input_path),
        judge_prompt="Judge it.",
        model_provider="openai",
        model_name="gpt-4o",
        prompt_column="prompt",
        reference_column="reference",
        use_reference=True,
        max_mismatches=2,
        output_path=str(output_path),
    )

    assert "metrics" in out
    assert "per_sample" in out
    assert "mismatches" in out
    assert "invalid_cases" in out

    assert out["metrics"]["samples"] == 3
    assert out["metrics"]["invalid_count"] == 1
    assert out["metrics"]["average_score"] == round((0.9 + 0.1 + 0.0) / 3, 4)

    assert len(out["mismatches"]) == 2
    assert out["mismatches"][0]["score"] == 0.0
    assert out["mismatches"][1]["score"] == 0.1

    assert len(out["invalid_cases"]) == 1
    assert out["invalid_cases"][0]["invalid"] is True

    assert len(out["per_sample"]) == 3
    row0 = out["per_sample"][0]
    assert "sample_id" in row0
    assert "judge.score" in row0
    assert "judge.invalid" in row0
    assert "judge.prompt" in row0
    assert "judge.model_answer" in row0
    assert "judge.reference" in row0

    assert output_path.exists()
    saved = _read_jsonl(output_path)
    assert len(saved) == 3

@pytest.mark.asyncio
async def test_evaluate_llm_judge_scale_01_offline_without_reference(monkeypatch, tmp_path: Path):
    """Verify use_reference=False skips reference in result rows (judge_example patched)."""
    input_path = tmp_path / "input.jsonl"
    _write_jsonl(
        input_path,
        [
            {"prompt": "P1", "model_answer": "A1", "reference": "R1"},
        ],
    )

    class FakePromptHandler:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("lib.evaluators.evaluate_llm_judge_scale_01.PromptHandler", FakePromptHandler)

    async def fake_judge_example(example, judge_prompt, prompt_handler, sample_id, **kwargs):
        return {
            "sample_id": int(sample_id),
            "prompt": example.get("prompt", ""),
            "model_answer": example.get("model_answer", ""),
            "score": 0.5,
            "invalid": False,
        }

    monkeypatch.setattr("lib.evaluators.evaluate_llm_judge_scale_01.judge_example", fake_judge_example)

    out = await evaluate_llm_judge_scale_01(
        input_path=str(input_path),
        judge_prompt="Judge it.",
        use_reference=False,
        max_mismatches=1,
        output_path=None,
    )

    assert out["metrics"]["samples"] == 1
    assert len(out["per_sample"]) == 1
    assert out["per_sample"][0]["judge.reference"] == ""