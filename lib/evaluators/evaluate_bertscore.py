import json
import argparse
from bert_score import BERTScorer
import os

def normalize(text: str) -> str:
    """Normalize text to a stripped string."""
    if text is None:
        return ""
    return str(text).strip()

def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")
    data = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {path}: {e}")
    if not data:
        raise ValueError("File is empty or contains no valid JSON lines.")
    return data

def extract_references_predictions(data, reference_column, model_answer_column):
    """Extract reference and prediction lists from dataset rows."""
    references, predictions = [], []
    for item in data:
        ref = normalize(item.get(reference_column, ""))
        pred = normalize(item.get(model_answer_column, ""))
        references.append(ref)
        predictions.append(pred)
    return references, predictions

def compute_bertscore(references, predictions, lang="en", model_type="bert-base-uncased"):
    """Compute BERTScore precision, recall, and F1 for given predictions and references."""
    if not references or not predictions:
        raise ValueError("Empty references or predictions list.")
    scorer = BERTScorer(lang=lang, model_type=model_type)
    P, R, F1 = scorer.score(predictions, references)
    return P, R, F1

def attach_scores_to_data(data, P, R, F1):
    """Attach per-record BERTScore metrics to dataset rows."""
    for idx, (p, r, f) in enumerate(zip(P, R, F1)):
        data[idx]["bertscore_precision"] = round(p.item(), 4)
        data[idx]["bertscore_recall"] = round(r.item(), 4)
        data[idx]["bertscore_f1"] = round(f.item(), 4)
    return data

def summarize_scores(P, R, F1, total):
    """Summarize average BERTScore metrics."""
    return {
        "bertscore_precision": round(P.mean().item(), 4),
        "bertscore_recall": round(R.mean().item(), 4),
        "bertscore_f1": round(F1.mean().item(), 4),
        "total": total
    }

def save_jsonl(data, output_path):
    """Save dataset with attached scores to JSONL file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    except Exception as e:
        raise IOError(f"Failed to save results to {output_path}: {e}")
    
def collect_bertscore_mismatches(
    references: list[str],
    predictions: list[str],
    f1_scores,
    max_mismatches: int = 10
) -> list[dict]:
    """Collect worst-scoring examples from BERTScore evaluation."""
    if max_mismatches < 0:
        raise ValueError("max_mismatches must be non-negative.")

    if not (len(references) == len(predictions) == len(f1_scores)):
        raise ValueError("Length mismatch between references, predictions, and f1_scores.")

    mismatches = []
    for ref, pred, f in zip(references, predictions, f1_scores):
        mismatches.append({
            "reference": ref,
            "predicted": pred,
            "f1": round(float(f), 4)
        })

    mismatches = sorted(mismatches, key=lambda x: x["f1"])
    return mismatches[:max_mismatches]

def evaluate_bertscore(
    input_path: str,
    reference_column: str = "summary",
    model_answer_column: str = "model_answer",
    max_mismatches: int = 10,
    output_path: str = None
):
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
    data = load_jsonl(input_path)
    references, predictions = extract_references_predictions(data, reference_column, model_answer_column)
    P, R, F1 = compute_bertscore(references, predictions)
    data = attach_scores_to_data(data, P, R, F1)
    result = summarize_scores(P, R, F1, len(data))

    result["mismatches"] = collect_bertscore_mismatches(references, predictions, F1, max_mismatches)

    if output_path:
        save_jsonl(data, output_path)

    return result