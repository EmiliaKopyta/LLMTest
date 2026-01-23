from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import pytest
from lib.data.DatasetLoader import DatasetLoader

# Helpers

def _write_csv(path: Path, content: str) -> None:
    """Create a CSV file at the given path. Prepare minimal test fixtures."""
    path.write_text(content, encoding="utf-8")

def _write_json(path: Path, data) -> None:
    """Create a JSON file at the given path using pandas."""
    pd.DataFrame(data).to_json(path, orient="records", force_ascii=False)

def _write_jsonl(path: Path, lines: list[dict]) -> None:
    """ Create a JSONL file at the given path."""
    text = "\n".join(json.dumps(obj, ensure_ascii=False) for obj in lines)
    path.write_text(text + "\n", encoding="utf-8")

# Unit tests: local files

def test_load_local_csv(tmp_path: Path):
    """
    Verify that DatasetLoader loads a local CSV file into a DataFrame.
    Returned object should be a pandas DataFrame with correct shape and columns, numeric values.
    """
    p = tmp_path / "sample.csv"
    _write_csv(p, "a,b\n1,2\n3,4\n")

    df = DatasetLoader.load(str(p))

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["a", "b"]
    assert df.iloc[0]["a"] == 1

def test_load_local_json(tmp_path: Path):
    """
    Verify that DatasetLoader loads a local JSON file.
    Checks proper DataFrame conversion, correct number of rows, expected columns.
    """
    p = tmp_path / "sample.json"
    _write_json(p, [{"x": 1, "y": "abc"}, {"x": 2, "y": "def"}])

    df = DatasetLoader.load(str(p))

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2
    assert set(df.columns) == {"x", "y"}
    assert df.iloc[1]["x"] == 2

def test_load_local_jsonl(tmp_path: Path):
    """
    Verify that DatasetLoader loads a local JSONL file.
    JSONL is expected to be parsed with 'lines=True' and produce one row per line.
    """
    p = tmp_path / "sample.jsonl"
    _write_jsonl(p, [{"x": 1}, {"x": 2}, {"x": 3}])

    df = DatasetLoader.load(str(p))

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 3
    assert "x" in df.columns
    assert df["x"].tolist() == [1, 2, 3]

def test_load_local_parquet(tmp_path: Path):
    """
    Verify that DatasetLoader loads a local Parquet file.
    Parquet requires 'pyarrow' or 'fastparquet', if it's not installed, this test is skipped.
    """
    p = tmp_path / "sample.parquet"
    df_in = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    try:
        df_in.to_parquet(p)
    except Exception as e:
        pytest.skip(f"Parquet support missing (pyarrow/fastparquet). Details: {e}")

    df_out = DatasetLoader.load(str(p))

    assert isinstance(df_out, pd.DataFrame)
    assert df_out.shape == (2, 2)
    assert df_out["a"].tolist() == [1, 2]
    assert df_out["b"].tolist() == ["x", "y"]

def test_load_from_folder_combines_supported_files(tmp_path: Path):
    """
    Verify recursive folder loading and concatenation.
    The folder structure is created dynamically:
        tmp_path/
         data1.csv
         nested/
          data2.jsonl
          ignore.txt

    Checks if supported files are detected recursively, unsupported files are ignored, 
    output DataFrame contains rows from both supported files.
    """
    (tmp_path / "nested").mkdir()

    csv_path = tmp_path / "data1.csv"
    _write_csv(csv_path, "a\n1\n2\n")

    jsonl_path = tmp_path / "nested" / "data2.jsonl"
    _write_jsonl(jsonl_path, [{"a": 3}, {"a": 4}])

    (tmp_path / "ignore.txt").write_text("ignore", encoding="utf-8")

    df = DatasetLoader.load(str(tmp_path))

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert "a" in df.columns
    assert sorted(df["a"].dropna().tolist()) == [1, 2, 3, 4]

def test_load_from_folder_raises_if_no_supported_files(tmp_path: Path):
    """Verify that folder loading fails if no supported dataset files are found."""
    (tmp_path / "readme.txt").write_text("hello", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        DatasetLoader.load(str(tmp_path))

def test_unsupported_format_raises_value_error(tmp_path: Path):
    """
    Verify that unsupported file extensions raise ValueError.
    Creates a `.txt` file and checks that DatasetLoader does not accept it.
    """
    p = tmp_path / "test.txt"
    p.write_text("dummy", encoding="utf-8")

    with pytest.raises(ValueError):
        DatasetLoader.load(str(p))

# Integration tests: Hugging Face

@pytest.mark.integration
def test_load_hf_csv_with_pandas_only():
    """
    Hugging Face CSV loading via pandas.
    This test depends on the environment (hf:// support, network access),
    it is skipped if the environment cannot access Hugging Face.
    """
    try:
        df = DatasetLoader.load("hf://datasets/domenicrosati/TruthfulQA/train.csv")
    except Exception as e:
        pytest.skip(f"hf:// not available in this environment: {e}")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty

@pytest.mark.integration
def test_load_hf_parquet_mmlu_with_pandas_only():
    """ Hugging Face Parquet loading via pandas. Skipped automatically if hf:// access is not available."""
    splits = {"test": "all/test-00000-of-00001.parquet"}

    try:
        df = DatasetLoader.load("hf://datasets/cais/mmlu/" + splits["test"])
    except Exception as e:
        pytest.skip(f"hf:// not available in this environment: {e}")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty