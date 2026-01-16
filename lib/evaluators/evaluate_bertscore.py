import os
import json
import pandas as pd
from bert_score import BERTScorer

def normalize(text: str) -> str:
    """Normalize text to a stripped string."""
    if text is None:
        return ""
    return str(text).strip()

def load_jsonl_df(path: str) -> pd.DataFrame:
    """Load a JSONL file into a DataFrame with basic validation."""
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

def save_jsonl_df(df: pd.DataFrame, output_path: str) -> None:
    """Save a DataFrame to JSONL (records, one JSON object per line)."""
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    except Exception as e:
        raise IOError(f"Failed to save results to {output_path}: {e}")

def validate_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    """Ensure required columns exist in the DataFrame."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {', '.join(missing)}")

def extract_references_predictions(df: pd.DataFrame, reference_column: str, model_answer_column: str) -> tuple[list[str], list[str]]:
    """Extract normalized reference/prediction lists from a DataFrame."""
    references = df[reference_column].apply(normalize).tolist()
    predictions = df[model_answer_column].apply(normalize).tolist()

    if not references or not predictions:
        raise ValueError("Empty references or predictions list.")
    if len(references) != len(predictions):
        raise ValueError("Length mismatch between references and predictions.")

    return references, predictions

def compute_bertscore(references: list[str], predictions: list[str], lang: str = "en", model_type: str = "bert-base-uncased"):
    """Compute BERTScore precision/recall/F1 tensors for given pairs."""
    scorer = BERTScorer(lang=lang, model_type=model_type)
    P, R, F1 = scorer.score(predictions, references)
    return P, R, F1

def build_per_sample_bertscore(references: list[str], predictions: list[str], P, R, F1, prefix: str = "bertscore") -> list[dict]:
    """Build per-sample, merge-friendly rows (keyed by sample_id)."""
    rows: list[dict] = []
    for sample_id, (p, r, f, ref, pred) in enumerate(zip(P, R, F1, references, predictions)):
        rows.append({
            "sample_id": int(sample_id),
            f"{prefix}.precision": round(p.item(), 4),
            f"{prefix}.recall": round(r.item(), 4),
            f"{prefix}.f1": round(f.item(), 4),
            f"{prefix}.reference": ref,
            f"{prefix}.predicted": pred,
        })
    return rows

def collect_bertscore_mismatches(per_sample: list[dict], max_mismatches: int = 10, f1_key: str = "bertscore.f1") -> list[dict]:
    """Collect worst-scoring examples by BERTScore F1."""
    if max_mismatches < 0:
        raise ValueError("max_mismatches must be non-negative.")
    if not per_sample:
        return []

    if f1_key not in per_sample[0]:
        raise KeyError(f"Missing key '{f1_key}' in per_sample rows.")

    sorted_rows = sorted(per_sample, key=lambda x: x[f1_key])
    return sorted_rows[:max_mismatches]

def summarize_bertscore(P, R, F1, total: int) -> dict:
    """Summarize average BERTScore metrics."""
    return {
        "bertscore_precision": round(P.mean().item(), 4),
        "bertscore_recall": round(R.mean().item(), 4),
        "bertscore_f1": round(F1.mean().item(), 4),
        "total": int(total),
    }

def attach_bertscore_to_df(df: pd.DataFrame, P, R, F1) -> pd.DataFrame:
    """Attach per-record BERTScore columns to the original DataFrame."""
    df_out = df.copy()
    df_out["bertscore_precision"] = [round(x.item(), 4) for x in P]
    df_out["bertscore_recall"] = [round(x.item(), 4) for x in R]
    df_out["bertscore_f1"] = [round(x.item(), 4) for x in F1]
    return df_out

def evaluate_bertscore(
    input_path: str,
    reference_column: str = "summary",
    model_answer_column: str = "model_answer",
    max_mismatches: int = 10,
    output_path: str | None = None,
    lang: str = "en",
    model_type: str = "bert-base-uncased"
) -> dict:
    """
    Evaluate BERTScore for a text generation task stored in JSONL format.

    This function loads a dataset of model outputs and references, computes
    BERTScore precision, recall, and F1 metrics, attaches per-record scores,
    and returns a summary dictionary. Optionally, it saves the enriched dataset
    with scores to a JSONL file.

    Parameters:
    input_path : str
        Path to the JSONL file containing model outputs and references.
    reference_column : str, default="summary"
        Column name containing the reference text.
    model_answer_column : str, default="model_answer"
        Column name containing the model-generated text.
    max_mismatches : int, default=10
        Maximum number of worst-scoring examples to include in the report.
    output_path : str, optional
        Path to save the evaluation results with per-record scores.

    Returns:
    dict
        A dictionary with average BERTScore precision, recall, F1, and total count.

    Example:
    >>> report = evaluate_bertscore("results.jsonl", "scored_results.jsonl")
    """
    df = load_jsonl_df(input_path)
    validate_required_columns(df, [reference_column, model_answer_column])

    references, predictions = extract_references_predictions(df, reference_column, model_answer_column)
    P, R, F1 = compute_bertscore(references, predictions, lang=lang, model_type=model_type)

    per_sample = build_per_sample_bertscore(references, predictions, P, R, F1, prefix="bertscore")
    mismatches = collect_bertscore_mismatches(per_sample, max_mismatches=max_mismatches, f1_key="bertscore.f1")
    metrics = summarize_bertscore(P, R, F1, total=len(df))

    if output_path:
        df_out = attach_bertscore_to_df(df, P, R, F1)
        save_jsonl_df(df_out, output_path)

    return {
        "metrics": metrics,
        "per_sample": per_sample,
        "mismatches": mismatches,
    }
