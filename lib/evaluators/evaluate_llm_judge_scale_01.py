import json
import os
import re
from lib.PromptHandler import PromptHandler

def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    if not data:
        raise ValueError("File is empty or contains no valid JSON lines.")
    return data

def parse_score(output: str) -> tuple[float, bool]:
    if not output:
        return 0.0, True
    text = output.strip().lower().replace(",", ".")
    try:
        return float(text), False
    except ValueError:
        match = re.search(r"(\d+(\.\d+)?)", text)
        if match:
            return float(match.group(1)), False
        return 0.0, True

def summarize_results(judged_results: list[dict], max_mismatches: int = 10) -> dict:
    if max_mismatches < 0:
        raise ValueError("max_mismatches must be non-negative.")
    total = len(judged_results)
    avg_score = round(sum(j["score"] for j in judged_results) / total, 4) if total > 0 else 0.0
    mismatches = sorted(judged_results, key=lambda x: x["score"])[:max_mismatches]
    invalid_cases = [j for j in judged_results if j.get("invalid")]
    return {
        "samples": total,
        "average_score": avg_score,
        "mismatches": mismatches,
        "invalid_cases": invalid_cases
    }

def save_jsonl(data: list[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

async def judge_example(
    example: dict,
    judge_prompt: str,
    prompt_handler: PromptHandler,
    prompt_column: str = "prompt",
    reference_column: str = "reference",
    use_reference: bool = True
) -> dict:
    """Evaluate a single example using the judge prompt, optionally with reference."""
    model_answer = example.get("model_answer", "").strip()
    prompt = example.get(prompt_column, "").strip()
    reference = example.get(reference_column, "").strip() if use_reference else None

    full_prompt = f"{judge_prompt.strip()}\n\nPrompt: {prompt}\n\n"
    if use_reference and reference:
        full_prompt += f"Reference: {reference}\n\n"
    full_prompt += f"Model Answer: {model_answer}\n\n"
    full_prompt += "Please respond with a single number between 0 and 1 representing the quality of the model answer."

    response = await prompt_handler.generate_response(full_prompt)
    score, invalid = parse_score(response.output)

    result = {
        "prompt": prompt,
        "model_answer": model_answer,
        "score": score,
        "invalid": invalid
    }
    if use_reference:
        result["reference"] = reference
    return result

async def evaluate_llm_judge_scale_01(
    input_path: str,
    judge_prompt: str,
    model_provider: str = "openai",
    model_name: str = "gpt-4o",
    prompt_column: str = "prompt",
    reference_column: str = "reference",
    use_reference: bool = True,
    max_mismatches: int = 10,
    output_path: str = None
) -> dict:
    """
    Evaluate model answers using an LLM judge on a 0–1 numeric scale.
    Works both with and without reference depending on `use_reference`.

    This function loads a dataset and evaluates each example using a judge prompt and a specified model provider,
    computes average scores, collects mismatches, and optionally saves results
    to a JSONL file.

    Parameters:
    input_path : str
        Path to the JSONL file containing model outputs and references.
    judge_prompt : str
        Instruction prompt for the judge model.
    model_provider : str, default="openai"
        Provider name for the judge model.
    model_name : str, default="gpt-4"
        Model name for the judge model.
    output_path : str, optional
        Path to save the evaluation results with per-record scores.
    prompt_column : str, default="prompt"
        Column name in the dataset containing the prompt text.
    reference_column : str, default="reference"
        Column name in the dataset containing the reference text.
    use_reference: bool, default = True
        Whether to include reference texts in the evaluation.
    max_mismatches : int, default=10
        Maximum number of worst-scoring examples to include in the summary.
    output_path : str, optional
        Path to save the evaluation results with per-record scores.

    Returns:
    dict
        A dictionary containing:
        - samples: number of evaluated examples
        - average_score: average judge score
        - mismatches: list of worst-scoring examples (up to max_mismatches)

    Note:
    If the judge model produces an invalid output (e.g., non-numeric text),
    the score is set to 0.0 as a fallback. Such cases are also collected
    in the `invalid_cases` list for inspection.

    Example:
    >>> summary = await evaluate_llm_judge_scale_01("results.jsonl", "Judge this answer")
    """
    results = load_jsonl(input_path)
    prompt_handler = PromptHandler(model_name=model_name, provider=model_provider, system_prompt="")

    judged_results = []
    for example in results:
        judged_results.append(await judge_example(
            example,
            judge_prompt,
            prompt_handler,
            prompt_column=prompt_column,
            reference_column=reference_column,
            use_reference=use_reference
        ))

    result_summary = summarize_results(judged_results, max_mismatches)

    if output_path:
        save_jsonl(judged_results, output_path)

    return result_summary