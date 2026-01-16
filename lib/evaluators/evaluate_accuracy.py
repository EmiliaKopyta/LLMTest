import ast
import os
import re
import json
import pandas as pd

def normalize(text: str, ignore_case: bool = True) -> str:
    """Normalize text to stripped string (optionally case-insensitive) with normalized whitespace."""
    if text is None:
        return ""

    s = str(text)
    s = s.replace("\u00a0", " ") #whitespace
    s = " ".join(s.strip().split())

    s = s.strip().strip('"').strip("'") #quotes

    if ignore_case:
        s = s.lower()
    return s

def strip_choice_prefix(text: str) -> str:
    """
    Remove leading multiple-choice prefixes like:
      "A. ", "b) ", "C: ", "d - "
    but do NOT touch numeric decimals like "0.242" (because it starts with a digit).
    """
    if text is None:
        return ""
    s = str(text).strip()
    return re.sub(r"^\s*([A-Da-d])\s*[\.\)\:\-]\s*", "", s).strip()

def parse_choices(raw):
    """Normalize the 'choices' column into a list of strings."""
    if isinstance(raw, list):
        return [str(c).strip() for c in raw]

    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(c).strip() for c in parsed]
        except Exception:
            pass
        cleaned = raw.strip("[]").replace("'", "").replace('"', "")
        return [c.strip() for c in cleaned.split() if c]

    return [str(raw).strip()]

def load_jsonl(path: str) -> pd.DataFrame:
    """Load JSONL file into a DataFrame with validation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")
    try:
        df = pd.read_json(path, lines=True)
    except ValueError as e:
        raise ValueError(f"Invalid JSONL format in {path}: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Encoding error in {path}: {e}")
    if df.empty:
        raise ValueError("File is empty.")
    return df

def evaluate_row_index_mode(row, answer_column, model_answer_column, choices_column, ignore_case: bool):
    """Evaluate a row assuming the correct answer is an index into choices."""
    choices = parse_choices(row[choices_column])
    raw_answer = row[answer_column]

    if not isinstance(raw_answer, int) and not (isinstance(raw_answer, str) and raw_answer.isdigit()):
        raise ValueError(f"Expected index in column '{answer_column}', got: {raw_answer}")

    idx = int(raw_answer)
    if idx < 0 or idx >= len(choices):
        raise ValueError(f"Index {idx} out of range for choices: {choices}")

    correct_answer = normalize(choices[idx], ignore_case=ignore_case)
    correct_letter = chr(65 + idx).lower()

    return correct_answer, correct_letter

def evaluate_row_text_mode(row, answer_column, ignore_case: bool):
    """Evaluate a row assuming the correct answer is a text string."""
    raw_answer = row[answer_column]
    correct_answer = normalize(raw_answer, ignore_case=ignore_case)
    correct_letter = None
    return correct_answer, correct_letter

def evaluate_row(row, answer_column, model_answer_column, choices_column, mode="index", ignore_case: bool = True):
    """Evaluate a single row using explicit mode: 'index' or 'text'."""
    if choices_column not in row or answer_column not in row or model_answer_column not in row:
        raise KeyError("Row is missing required fields.")

    if mode == "index":
        correct_answer, correct_letter = evaluate_row_index_mode(
            row, answer_column, model_answer_column, choices_column, ignore_case=ignore_case
        )
    elif mode == "text":
        correct_answer, correct_letter = evaluate_row_text_mode(
            row, answer_column, ignore_case=ignore_case
        )
    else:
        raise ValueError("mode must be 'index' or 'text'")

    raw_model_answer = row[model_answer_column]
    raw_model_answer_str = "" if raw_model_answer is None else str(raw_model_answer).strip()

    model_answer_letter = normalize(raw_model_answer_str, ignore_case=True) #accepting letter answer, (ex. A) before stripping prefix
    is_letter_only = bool(re.fullmatch(r"[a-d]", model_answer_letter))

    cleaned_model_answer = strip_choice_prefix(raw_model_answer_str) #stripping prefix
    model_answer = normalize(cleaned_model_answer, ignore_case=ignore_case)

    is_correct = (
        model_answer == correct_answer or
        (correct_letter and (is_letter_only and model_answer_letter == correct_letter))
    )

    per_sample = {
        "expected": correct_answer,
        "predicted_raw": raw_model_answer,
        "predicted_norm": model_answer,
        "correct": int(is_correct),
        "expected_letter": correct_letter,
        "predicted_letter": model_answer_letter if is_letter_only else None,
    }
    return is_correct, per_sample

def build_per_sample_accuracy(df, answer_column, question_column, model_answer_column, choices_column, mode, ignore_case):
    """Build per-sample accuracy results with correctness and metadata."""
    per_sample = []
    total = 0
    correct = 0

    for sample_id, row in df.iterrows():
        is_correct, sample = evaluate_row(
            row, answer_column, model_answer_column, choices_column, mode=mode, ignore_case=ignore_case
        )
        sample["sample_id"] = sample_id

        sample_out = {
            "sample_id": sample_id,
            "accuracy.question": row[question_column],
            "accuracy.correct": sample["correct"],
            "accuracy.expected": sample["expected"],
            "accuracy.predicted_raw": sample["predicted_raw"],
            "accuracy.predicted_norm": sample["predicted_norm"],
        }
        per_sample.append(sample_out)
        total += 1
        correct += int(is_correct)

    return per_sample, total, correct

def collect_mismatches(per_sample: list[dict], max_mismatches: int) -> list[dict]:
    """Select worst-scoring examples based on incorrect predictions."""
    if max_mismatches < 0:
        raise ValueError("max_mismatches must be non-negative.")
    return [s for s in per_sample if s["accuracy.correct"] == 0][:max_mismatches]

def summarize_results(total, correct):
    """Build the final evaluation report."""
    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
    }

def save_report_jsonl(metrics: dict, output_path: str):
    """Save evaluation metrics to JSONL file."""
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for key, value in metrics.items():
                f.write(json.dumps({key: value}, ensure_ascii=False) + "\n")
    except Exception as e:
        raise IOError(f"Failed to save results to {output_path}: {e}")

def evaluate_accuracy(
    input_path: str,
    answer_column: str = "answer",
    question_column: str = "question",
    model_answer_column: str = "model_answer",
    choices_column: str = "choices",
    mode: str = "index",
    max_mismatches: int = 10,
    output_path: str = None,
    ignore_case: bool = True
):
    """
    Evaluate accuracy for a multiple-choice task stored in JSONL format.

    Parameters:
    input_path : str
        Path to the JSONL file with model outputs.
    answer_column : str, default="answer"
        Column name containing the correct answer (index or string).
    model_answer_column : str, default="model_answer"
        Column name containing the model's predicted answer.
    choices_column : str, default="choices"
        Column name containing the list of possible choices.
    mode : str, default="index"
        Determines how the correct answer should be interpreted. Supported values:
        - "index": the value in `answer_column` is treated as an integer index pointing to the correct option inside `choices_column`.
        - "text": the value in `answer_column` is treated as a literal text string and compared directly to the model prediction.
    max_mismatches : int, default=10
        Maximum number of mismatches to include in the report.
    output_path : str, optional
        Path to save the evaluation summary as JSONL.
    ignore_case : bool, default=True
        If True, comparison is case-insensitive (recommended for units like GeV vs gev).

    Returns:
    dict
        Dictionary with keys:
        - `accuracy`: float, overall accuracy
        - `total`: int, number of samples
        - `correct`: int, number of correct predictions
        - `incorrect`: int, number of incorrect predictions
        - `mismatches`: list of dicts with up to `max_mismatches` examples
    """
    df = load_jsonl(input_path)

    required_cols = [answer_column, model_answer_column, choices_column, question_column]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    per_sample, total, correct = build_per_sample_accuracy(
        df, answer_column, question_column, model_answer_column,
        choices_column, mode, ignore_case
    )
    mismatches = collect_mismatches(per_sample, max_mismatches)
    metrics = summarize_results(total, correct, mismatches)

    if output_path:
        save_report_jsonl(metrics, output_path)

    return {
        "metrics": metrics,
        "per_sample": per_sample,
        "mismatches": mismatches,
    }
