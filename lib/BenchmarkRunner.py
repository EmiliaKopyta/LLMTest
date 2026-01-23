import os
import json
import asyncio
import logging
import pandas as pd
from datetime import datetime
from typing import Callable

from lib.TestRunner import TestRunner

logger = logging.getLogger(__name__)

class ModelSpec:
    """Defines a single model used in the benchmark."""
    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name

class EvaluationSpec:
    """Defines a named evaluation function applied to model outputs."""
    def __init__(self, name: str, evaluator: Callable[[str], dict]):
        self.name = name
        self.evaluator = evaluator

def load_results_df(results_path: str, test_name: str, model: ModelSpec, output_format: str = "jsonl") -> pd.DataFrame:
    """Load model results and enrich them with benchmark metadata."""
    ext = os.path.splitext(results_path)[1].lower()

    if output_format.lower() == "csv" or ext == ".csv":
        df = pd.read_csv(results_path)
    else:
        df = pd.read_json(results_path, lines=True)

    df["benchmark"] = test_name
    df["model_provider"] = model.provider
    df["model_name"] = model.model_name
    df["model"] = f"{model.provider}:{model.model_name}"
    df["sample_id"] = df.index
    return df

def attach_per_sample(df: pd.DataFrame, per_sample: list[dict]) -> pd.DataFrame:
    """Helper method used when attaching evaluation scores to the result dataset."""
    per_df = pd.DataFrame(per_sample)

    if "sample_id" not in per_df.columns:
        raise ValueError("per_sample must contain 'sample_id'")

    return df.merge(per_df, on="sample_id", how="left", validate="one_to_one")

async def run_evaluation(eval_spec: EvaluationSpec, results_path: str, test_name: str, model: ModelSpec, system_prompt: str | list[str]) -> dict:
    """Run a single evaluation on a model result file."""
    logger.info("Evaluating %s:%s with %s", model.provider, model.model_name, eval_spec.name)
    
    result = eval_spec.evaluator(results_path)
    result = await result if asyncio.iscoroutine(result) else result

    metrics = result["metrics"]
    per_sample = result["per_sample"]
    mismatches = result.get("mismatches", [])

    report_row = {
        "benchmark": test_name,
        "model": f"{model.provider}:{model.model_name}",
        "provider": model.provider,
        "model_name": model.model_name,
        "system_prompt": system_prompt,
        "evaluation": eval_spec.name,
        "metrics": metrics,
        "mismatches": mismatches
    }
    return report_row, per_sample

def _ext_from_format(output_format: str) -> str:
    if output_format not in ("jsonl", "json", "csv"):
        raise ValueError("output_format must be 'jsonl', 'json', or 'csv'")
    return output_format

def resolve_benchmark_path(benchmark_name: str, output_format: str, kind: str, base_dir: str, timestamp: str, output_path: str | None = None) -> str:
    """Return the output file path for a benchmark dataset or report."""
    ext = _ext_from_format(output_format)

    if output_path:
        return output_path

    folder = os.path.join(base_dir, benchmark_name)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{kind}_{timestamp}.{ext}")

def save_benchmark_dataset(df: pd.DataFrame, output_path: str, output_format: str) -> str:
    """Save the combined benchmark dataset."""
    if output_format == "jsonl":
        df.to_json(output_path, orient="records", lines=True)
    elif output_format == "json":
        df.to_json(output_path, orient="records", indent=2)
    elif output_format == "csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError("output_format must be 'jsonl', 'json', or 'csv'")

    logger.info("Benchmark dataset saved to: %s", output_path)
    return output_path

def save_benchmark_report(report_rows: list[dict], report_path: str, output_format: str) -> str:
    """Save the benchmark evaluation report."""
    if output_format == "jsonl":
        with open(report_path, "w", encoding="utf-8") as f:
            for row in report_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif output_format == "json":
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_rows, f, indent=2, ensure_ascii=False)
    elif output_format == "csv":
        pd.DataFrame(report_rows).to_csv(report_path, index=False)
    else:
        raise ValueError("output_format must be 'jsonl', 'json', or 'csv'")

    logger.info("Benchmark report saved to: %s", report_path)
    return report_path

class BenchmarkRunner:
    """
    Orchestrates running multiple TestRunners on multiple models,
    aggregates their outputs into a single benchmark dataset,
    and runs multiple evaluation functions on the generated results.

    Outputs:
    - benchmark_dataset.(jsonl|csv)
    - benchmark_report.jsonl
    """

    def __init__(
        self,
        test_name: str,
        dataset_path: str,
        prompt_builder: Callable,
        models: list[ModelSpec],
        evaluations: list[EvaluationSpec],
        system_prompt: str | list[str] = "",
        dataset_splits: list[str] | None = None,
        output_dir: str = "benchmarks",
        output_format: str = "jsonl",
        report_format: str = "json",
        max_concurrency: int = 10
    ):
        self.test_name = test_name
        self.dataset_path = dataset_path
        self.prompt_builder = prompt_builder
        self.models = models
        self.evaluations = evaluations
        self.system_prompt = system_prompt
        self.dataset_splits = dataset_splits
        self.output_dir = output_dir
        self.output_format = output_format
        self.report_format = report_format
        self.max_concurrency = max_concurrency

    async def run(self, selection: str | None = None) -> dict:
        """Execute the full benchmark pipeline."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        all_dfs: list[pd.DataFrame] = []
        report_rows: list[dict] = []

        for model in self.models:
            logger.info("Running benchmark for model %s:%s", model.provider, model.model_name)

            runner = TestRunner(
                test_name=self.test_name,
                test_dataset_path=self.dataset_path,
                model_provider=model.provider,
                model_name=model.model_name,
                prompt_builder=self.prompt_builder,
                system_prompt=self.system_prompt,
                dataset_splits=self.dataset_splits,
                output_format=self.output_format,
                max_concurrency=self.max_concurrency
            )
            results_path = await runner.run_model(selection=selection)

            df = load_results_df(results_path, self.test_name, model, output_format=self.output_format)

            for eval_spec in self.evaluations:
                report_row, per_sample = await run_evaluation(eval_spec, results_path, self.test_name, model, self.system_prompt)
                report_rows.append(report_row)

                df = attach_per_sample(df, per_sample)

            all_dfs.append(df)

        benchmark_df = pd.concat(all_dfs, ignore_index=True)

        dataset_path = resolve_benchmark_path(benchmark_name=self.test_name,output_format=self.output_format,kind="dataset",base_dir=self.output_dir,timestamp=timestamp,output_path=None)
        report_path = resolve_benchmark_path(benchmark_name=self.test_name,output_format=self.report_format,kind="report",base_dir=self.output_dir,timestamp=timestamp,output_path=None)

        save_benchmark_dataset(benchmark_df, dataset_path, self.output_format)
        save_benchmark_report(report_rows, report_path, self.report_format)

        logger.info("Benchmark completed successfully")

        return {
            "benchmark": self.test_name,
            "dataset_path": dataset_path,
            "report_path": report_path,
            "models": [f"{m.provider}:{m.model_name}" for m in self.models],
            "selection": selection,
            "timestamp": timestamp
        }

