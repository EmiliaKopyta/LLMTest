import json
import argparse
from rouge_score import rouge_scorer
import os

def normalize(text: str, case_sensitive: bool = True) -> str:
    """Normalize text to lowercase stripped string."""
    if text is None:
        return ""
    text = str(text).strip()
    return text if case_sensitive else text.lower()

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

def compute_rouge_scores(data, reference_column="summary", model_answer_column="model_answer", case_sensitive: bool = True):
    """Compute ROUGE scores for each record and return lists of F1 scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_f1_scores, rouge2_f1_scores, rougel_f1_scores = [], [], []

    for item in data:
        reference = normalize(item.get(reference_column, ""), case_sensitive=case_sensitive)
        predicted = normalize(item.get(model_answer_column, ""), case_sensitive=case_sensitive)
        if not reference or not predicted:
            continue
        scores = scorer.score(reference, predicted)
        rouge1_f1_scores.append(scores["rouge1"].fmeasure)
        rouge2_f1_scores.append(scores["rouge2"].fmeasure)
        rougel_f1_scores.append(scores["rougeL"].fmeasure)

    return rouge1_f1_scores, rouge2_f1_scores, rougel_f1_scores

def summarize_rouge(rouge1_f1_scores, rouge2_f1_scores, rougel_f1_scores):
    """Summarize average ROUGE scores into a result dictionary."""
    total = len(rouge1_f1_scores)
    return {
        "samples": total,
        "rouge1_f1_avg": round(sum(rouge1_f1_scores) / total, 4) if total > 0 else 0,
        "rouge2_f1_avg": round(sum(rouge2_f1_scores) / total, 4) if total > 0 else 0,
        "rougeL_f1_avg": round(sum(rougel_f1_scores) / total, 4) if total > 0 else 0
    }

def save_report_jsonl(result: dict, output_path: str):
    """Save evaluation report dictionary to a JSONL file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    except Exception as e:
        raise IOError(f"Failed to save results to {output_path}: {e}")
    
def collect_rouge_mismatches(
    data: list[dict],
    reference_column: str,
    model_answer_column: str,
    rougeL_f1_scores,
    max_mismatches: int = 10
) -> list[dict]:
    """Collect worst-scoring examples from ROUGE evaluation."""
    if max_mismatches < 0:
        raise ValueError("max_mismatches must be non-negative.")

    if len(data) != len(rougeL_f1_scores):
        raise ValueError("Length mismatch between data and rougeL_f1_scores.")

    mismatches = []
    for item, f in zip(data, rougeL_f1_scores):
        mismatches.append({
            "reference": item.get(reference_column, ""),
            "predicted": item.get(model_answer_column, ""),
            "rougeL_f1": round(float(f), 4)
        })

    mismatches = sorted(mismatches, key=lambda x: x["rougeL_f1"])
    return mismatches[:max_mismatches]

def evaluate_rouge(
    input_path: str,
    reference_column: str = "summary",
    model_answer_column: str = "model_answer",
    case_sensitive: bool = False,
    max_mismatches: int = 10,
    output_path: str = None
):
    """
    Evaluate ROUGE similarity between model outputs and references stored in JSONL format.

    This function loads a dataset of model outputs and reference summaries, computes
    ROUGE-1, ROUGE-2, and ROUGE-L F1 scores, and returns a summary dictionary.
    Optionally, the summary can be saved to a JSONL file.

    Parameters:
    input_path : str
        Path to the JSONL file containing model outputs and references.
    reference_column : str, default="summary"
        Column name containing the reference text.
    model_answer_column : str, default="model_answer"
        Column name containing the model-generated text.
    case_sensitive : bool, default=False
        Controls whether ROUGE should treat uppercase and lowercase characters as distinct.
    max_mismatches : int, default=10
        Maximum number of worst-scoring examples to include in the report.
    output_path : str, optional
        Path to save the evaluation summary as JSONL.

    Returns:
    dict
        A dictionary containing:
        - samples: number of evaluated pairs
        - rouge1_f1_avg: average ROUGE-1 F1 score
        - rouge2_f1_avg: average ROUGE-2 F1 score
        - rougeL_f1_avg: average ROUGE-L F1 score

    Example:
    >>> report = evaluate_rouge("results.jsonl", "rouge_report.jsonl")
    """
    data = load_jsonl(input_path)
    rouge1_f1_scores, rouge2_f1_scores, rougel_f1_scores = compute_rouge_scores(
        data, reference_column, model_answer_column, case_sensitive=case_sensitive
    )
    result = summarize_rouge(rouge1_f1_scores, rouge2_f1_scores, rougel_f1_scores)

    result["mismatches"] = collect_rouge_mismatches(data, reference_column, model_answer_column, rougel_f1_scores, max_mismatches)

    if output_path:
        save_report_jsonl(result, output_path)

    return result