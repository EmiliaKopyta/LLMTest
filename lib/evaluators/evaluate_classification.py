import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import json

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

def compute_stats(df, label_column="label", model_answer_column="model_answer"):
    """Compute per-class TP/FP/FN and precision/recall/F1."""
    if label_column not in df.columns or model_answer_column not in df.columns:
        raise KeyError(f"Missing required columns: {label_column}, {model_answer_column}")

    stats = {}
    correct = 0
    total = len(df)
    if total == 0:
        raise ValueError("DataFrame is empty, cannot compute stats.")

    for _, row in df.iterrows():
        true_label = str(row[label_column]).strip().lower()
        pred_label = str(row[model_answer_column]).strip().lower()

        if pred_label == true_label:
            correct += 1
            stats.setdefault(true_label, {"tp":0,"fp":0,"fn":0})["tp"] += 1
        else:
            stats.setdefault(pred_label, {"tp":0,"fp":0,"fn":0})["fp"] += 1
            stats.setdefault(true_label, {"tp":0,"fp":0,"fn":0})["fn"] += 1

    precisions, recalls, f1s = [], [], []
    for cls, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        precision = tp/(tp+fp) if tp+fp>0 else 0
        recall = tp/(tp+fn) if tp+fn>0 else 0
        f1 = 2*precision*recall/(precision+recall) if precision+recall>0 else 0
        precisions.append(precision); recalls.append(recall); f1s.append(f1)
        stats[cls].update({
            "precision": round(precision,4),
            "recall": round(recall,4),
            "f1": round(f1,4)
        })

    return {
        "accuracy": round(correct/total,4),
        "total": total,
        "correct": correct,
        "incorrect": total-correct,
        "macro_precision": round(sum(precisions)/len(precisions),4),
        "macro_recall": round(sum(recalls)/len(recalls),4),
        "macro_f1": round(sum(f1s)/len(f1s),4),
        "per_class": stats
    }

def build_per_sample(df, label_column, model_answer_column) -> list[dict]:
    """Build per-sample classification rows with sample_id and correctness."""
    per_sample = []

    for i, row in df.iterrows():
        true_label = str(row[label_column]).strip().lower()
        pred_label = str(row[model_answer_column]).strip().lower()

        per_sample.append({
            "sample_id": i,
            "classification.text": row.get("text", ""),
            "classification.expected": true_label,
            "classification.predicted": pred_label,
            "classification.correct": int(pred_label == true_label)
        })

    return per_sample

def collect_mismatches(df, label_column="label", model_answer_column="model_answer", max_mismatches=10):
    """Collect mismatched predictions for inspection."""
    if label_column not in df.columns or model_answer_column not in df.columns:
        raise KeyError(f"Missing required columns: {label_column}, {model_answer_column}")
    if max_mismatches < 0:
        raise ValueError("max_mismatches must be non-negative.")

    mismatches = []
    for _, row in df.iterrows():
        true_label = str(row[label_column]).strip().lower()
        pred_label = str(row[model_answer_column]).strip().lower()
        if pred_label != true_label:
            mismatches.append({
                "question": row.get("text",""),
                "expected": true_label,
                "predicted": row[model_answer_column]
            })
    return mismatches[:max_mismatches]

def build_confusion_matrix(df, label_column, model_answer_column) -> dict:
    """Build a confusion matrix (true_label -> predicted_label counts) from predictions."""
    matrix = {}
    for _, row in df.iterrows():
        true_label = str(row[label_column]).strip().lower()
        pred_label = str(row[model_answer_column]).strip().lower()
        matrix.setdefault(true_label, {})
        matrix[true_label][pred_label] = matrix[true_label].get(pred_label, 0) + 1
    return matrix

def plot_confusion_matrix(confusion: dict):
    """Plot a confusion matrix heatmap from a dict of counts."""

    if not confusion:
        raise ValueError("Confusion matrix is empty, cannot plot.")
    df_conf = pd.DataFrame(confusion).fillna(0)
    if df_conf.empty:
        raise ValueError("Confusion matrix DataFrame is empty.")
    try:
        df_conf = pd.DataFrame.from_dict(confusion, orient="index").fillna(0).astype(int)
    except ValueError as e:
        raise ValueError(f"Confusion matrix contains non-numeric values: {e}")

    sns.heatmap(df_conf, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

def save_jsonl(result: dict, output_path: str):
    """Save evaluation results to a JSONL file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    except Exception as e:
        raise IOError(f"Failed to save results to {output_path}: {e}")

def evaluate_classification(
    input_path: str,
    label_column: str = "label",
    model_answer_column: str = "model_answer",
    include_confusion: bool = False,
    max_mismatches: int = 10,
    output_path: str = None
):
    """
    Evaluate classification results stored in a JSONL file.

    This function compares true labels against model predictions and computes
    standard classification metrics such as accuracy, macro precision, macro recall,
    and macro F1-score. It also provides per-class statistics and a list of sample
    mismatches for inspection. Optionally, it can generate a confusion matrix
    visualization using seaborn.

    Parameters:
    input_path : str
        Path to the JSONL file containing evaluation results.
    label_column : str, default="label"
        Column name holding the true labels in the dataset.
    model_answer_column : str, default="model_answer"
        Column name holding the model's predicted labels.
    include_confusion : bool, default=False
        Whether to compute and display a confusion matrix heatmap.
    max_mismatches : int, default=10
        Maximum number of mismatched examples to include in the output.
    output_path : str, optional
        Path to save the evaluation summary as JSON.

    Returns:
    dict
        A dictionary containing:
        - accuracy, total, correct, incorrect
        - macro_precision, macro_recall, macro_f1
        - per_class metrics (precision, recall, F1 for each label)
        - mismatches (up to `max_mismatches`)
        - confusion_matrix (if include_confusion=True)

    Example:
    >>> report = evaluate_classification_jsonl("results.jsonl", include_confusion=True)
    """
    df = load_jsonl(input_path)
    
    stats = compute_stats(df, label_column, model_answer_column)
    per_sample = build_per_sample(df, label_column, model_answer_column)
    mismatches = [s for s in per_sample if s["classification.correct"] == 0][:max_mismatches]

    metrics = {
        "accuracy": stats["accuracy"],
        "total": stats["total"],
        "correct": stats["correct"],
        "incorrect": stats["incorrect"],
        "macro_precision": stats["macro_precision"],
        "macro_recall": stats["macro_recall"],
        "macro_f1": stats["macro_f1"],
        "per_class": stats["per_class"]
    }

    if include_confusion:
        confusion = build_confusion_matrix(df, label_column, model_answer_column)
        metrics["confusion_matrix"] = confusion
        plot_confusion_matrix(confusion)

    if output_path:
        save_jsonl(per_sample, output_path)

    return {
        "metrics": metrics,
        "per_sample": per_sample,
        "mismatches": mismatches
    }
