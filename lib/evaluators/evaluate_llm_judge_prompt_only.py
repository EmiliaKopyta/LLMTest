import json
import os
from lib.PromptHandler import PromptHandler

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

def _is_float(value: str) -> bool:
    try:
        float(str(value).replace(",", "."))
        return True
    except Exception:
        return False
    
def build_judge_prompt(judge_prompt: str, prompt: str, model_answer: str) -> str:
    """Build full prompt to send to judge model."""
    return (
        f"{judge_prompt.strip()}\n\n"
        f"Prompt:\n{prompt.strip()}\n\n"
        f"Model Answer:\n{model_answer.strip()}\n\n"
        f"Please respond according to the criteria above."
    )

async def run_llm_judgment(full_prompt: str, prompt_handler: PromptHandler) -> str:
    """Send prompt to LLM and return stripped response."""
    response = await prompt_handler.generate_response(full_prompt)
    return (response.output or "").strip()

def collect_mismatches(judged_results: list[dict], max_mismatches: int = 10) -> list[dict]:
    """
    Collect mismatches (worst or most notable cases) from judged results.
    If judgments are numeric, sort ascending. Otherwise, take the first N.
    """
    if max_mismatches < 0:
        raise ValueError("max_mismatches must be non-negative.")

    numeric_results = [j for j in judged_results if _is_float(j.get("judge.judgment"))]
    if numeric_results:
        sorted_results = sorted(
            numeric_results,
            key=lambda x: float(str(x["judge.judgment"]).replace(",", "."))
        )
    else:
        sorted_results = judged_results

    return sorted_results[:max_mismatches]

def save_jsonl(rows: list[dict] | dict, output_path: str) -> None:
    """Save dataset with attached judgments to JSONL file."""
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            if isinstance(rows, list):
                for item in rows:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            elif isinstance(rows, dict):
                f.write(json.dumps(rows, ensure_ascii=False) + "\n")
            else:
                raise TypeError(f"Expected list[dict] or dict, got {type(rows)}")

    except Exception as e:
        raise IOError(f"Failed to save results to {output_path}: {e}")

async def evaluate_llm_judge_prompt_only(
    input_path: str,
    judge_prompt: str,
    model_provider: str = "openai",
    model_name: str = "gpt-4o",
    prompt_column: str = "prompt",
    max_mismatches: int = 10,
    output_path: str = None
) -> dict:
    """
    Evaluate model answers using an LLM judge (prompt-only, user-defined format).
    The judge_prompt defines the evaluation criteria and output format.
    Returns a unified schema with: prompt, model_answer, judgment, and mismatches.

    Results are heuristic, subjective, and may vary depending on the model
    and prompt. Use at your own risk. The output type and format
    of the judge’s response are NOT enforced.

    Parameters:
    input_path : str
        Path to the JSONL file containing prompts and model answers.
    judge_prompt : str
        Instruction prompt for the judge model (defines evaluation criteria and output format).
    model_provider : str, default="openai"
        Provider name for the judge model.
    model_name : str, default="gpt-4"
        Model name for the judge model.
    prompt_column : str, default="prompt"
        Column name containing the prompt (e.g., "dialogue").
    max_mismatches : int, default=10
        Maximum number of mismatched examples to include in the summary.
    output_path : str, optional
        Path to save the per-record judgments JSONL.

    Returns:
    dict
        A dictionary containing:
        - samples : number of evaluated examples.
        - judgments : list of per‑example results, each with:
          * prompt : str - the input prompt
          * model_answer : str - the model’s answer
          * judgment : str - the judge’s evaluation
        - mismatches : subset of worst‑scoring examples (up to `max_mismatches`)

    Example:
    >>> summary = await evaluate_llm_judge_prompt_only(input_path=input_path,judge_prompt=judge_prompt)
    """
    data = load_jsonl(input_path)
    prompt_handler = PromptHandler(model_name=model_name, provider=model_provider, system_prompt="")

    per_sample = []

    for i, example in enumerate(data):
        prompt = (example.get(prompt_column) or "N/A")
        model_answer = (example.get("model_answer") or "N/A")
        
        full_prompt = build_judge_prompt(judge_prompt, prompt, model_answer)
        judgment = await run_llm_judgment(full_prompt, prompt_handler)

        per_sample.append({
            "sample_id": i,
            "judge.prompt": prompt.strip(),
            "judge.model_answer": model_answer.strip(),
            "judge.judgment": judgment
        })

    mismatches = collect_mismatches(per_sample, max_mismatches=max_mismatches)

    metrics = {
        "samples": len(per_sample),
        "text_judgments": sum(1 for x in per_sample if x["judge.judgment"])
    }

    if output_path:
        save_jsonl(per_sample, output_path)

    return {
        "metrics": metrics,
        "per_sample": per_sample,
        "mismatches": mismatches
    }
