import ast
import os
import json
import pandas as pd


def normalize(text: str) -> str:
    """Normalize text to lowercase stripped string."""
    if text is None:
        return ""
    return str(text).strip().lower()


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

def evaluate_row_index_mode(row, answer_column, model_answer_column, choices_column):
    """Evaluate a row assuming the correct answer is an index into choices."""
    choices = parse_choices(row[choices_column])
    raw_answer = row[answer_column]

    if not isinstance(raw_answer, int) and not (isinstance(raw_answer, str) and raw_answer.isdigit()):
        raise ValueError(f"Expected index in column '{answer_column}', got: {raw_answer}")

    idx = int(raw_answer)
    if idx < 0 or idx >= len(choices):
        raise ValueError(f"Index {idx} out of range for choices: {choices}")

    correct_answer = normalize(choices[idx])
    correct_letter = chr(65 + idx).lower()

    return correct_answer, correct_letter

def evaluate_row_text_mode(row, answer_column):
    """Evaluate a row assuming the correct answer is a text string."""
    raw_answer = row[answer_column]
    correct_answer = normalize(raw_answer)
    correct_letter = None
    return correct_answer, correct_letter

def evaluate_row(row, answer_column, model_answer_column, choices_column, mode="index"):
    """Evaluate a single row using explicit mode: 'index' or 'text'."""
    if choices_column not in row or answer_column not in row or model_answer_column not in row:
        raise KeyError("Row is missing required fields.")

    if mode == "index":
        correct_answer, correct_letter = evaluate_row_index_mode(
            row, answer_column, model_answer_column, choices_column
        )
    elif mode == "text":
        correct_answer, correct_letter = evaluate_row_text_mode(
            row, answer_column
        )
    else:
        raise ValueError("mode must be 'index' or 'text'")

    model_answer = normalize(row[model_answer_column])
    if len(model_answer) > 2 and model_answer[1] == ".":
        model_answer = model_answer[2:].strip()

    is_correct = (
        model_answer == correct_answer or
        (correct_letter and model_answer.startswith(correct_letter))
    )

    mismatch = None
    if not is_correct:
        mismatch = {
            "question": row.get("question", ""),
            "expected": correct_answer,
            "predicted": row[model_answer_column],
        }

    return is_correct, mismatch

def summarize_results(total, correct, mismatches):
    """Build the final evaluation report."""
    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "mismatches": mismatches,
    }

def save_report_jsonl(result: dict, output_path: str):
    """Save evaluation report dictionary to a JSONL file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for key, value in result.items():
                f.write(json.dumps({key: value}, ensure_ascii=False) + "\n")
    except Exception as e:
        raise IOError(f"Failed to save results to {output_path}: {e}")

def evaluate_accuracy(
    input_path: str,
    answer_column: str = "answer",
    model_answer_column: str = "model_answer",
    choices_column: str = "choices",
    mode = "index",
    max_mismatches: int = 10,
    output_path: str = None
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

    Returns:
    dict
        Dictionary with keys:
        - `accuracy`: float, overall accuracy
        - `total`: int, number of samples
        - `correct`: int, number of correct predictions
        - `incorrect`: int, number of incorrect predictions
        - `mismatches`: list of dicts with up to `max_mismatches` examples

    Example:
    >>> report = evaluate_accuracy_jsonl(input_path)
    """
    df = load_jsonl(input_path)

    required_cols = [answer_column, model_answer_column, choices_column]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    total = 0
    correct = 0
    mismatches = []

    for _, row in df.iterrows():
        is_correct, mismatch = evaluate_row(row, answer_column, model_answer_column, choices_column, mode=mode)
        total += 1
        if is_correct:
            correct += 1
        else:
            mismatches.append(mismatch)

    result = summarize_results(total, correct, mismatches)
    if max_mismatches < 0:
        raise ValueError("max_mismatches must be non-negative.")
    result["mismatches"] = mismatches[:max_mismatches]

    if output_path:
        save_report_jsonl(result, output_path)

    return result
