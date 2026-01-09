import os
import asyncio
import pandas as pd
from datetime import datetime
from typing import Any
from lib.PromptHandler import PromptHandler
from lib.data.DatasetLoader import DatasetLoader
import logging

logger = logging.getLogger(__name__)

async def generate_single_response(handler, prompt, idx: int):
    """Generates response for a single prompt."""
    try:
        response = await handler.generate_response(prompt)
        return response.output
    except Exception as e:
        logger.warning("Skipping row %s due to error: %s", idx, e, exc_info=True)
        return None
    
def validate_prompt_output(prompt: Any, idx: int | None = None) -> None:
    """Validate that prompt_builder returned a supported prompt type."""
    if prompt is None:
        return
    if isinstance(prompt, str):
        return
    if isinstance(prompt, (list, tuple)):
        return

    where = f" (row {idx})" if idx is not None else ""
    raise TypeError(
        f"prompt_builder must return str | Sequence[UserContent] | None{where}, "
        f"got: {type(prompt)}"
    )

async def generate_sequential(df: pd.DataFrame, prompt_builder, handler):
    """Generates responses sequentially."""
    answers = []
    for idx, row in df.iterrows():
        prompt = prompt_builder(row)
        validate_prompt_output(prompt, idx)
        output = await generate_single_response(handler, prompt, idx)
        answers.append(output)
    return answers

async def generate_concurrent(df: pd.DataFrame, prompt_builder, handler, max_concurrency: int):
    """Generates responses concurrently."""
    sem = asyncio.Semaphore(max_concurrency)

    async def safe_generate(idx, row):
        async with sem:
            prompt = prompt_builder(row)
            validate_prompt_output(prompt, idx)
            return await generate_single_response(handler, prompt, idx)

    tasks = [
        safe_generate(idx, row)
        for idx, row in df.iterrows()
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=False)
    return responses

async def generate_answers(df: pd.DataFrame, prompt_builder, handler, max_concurrency: int):
    """Router for choosing the type of response generation."""
    if max_concurrency == 0:
        logger.info("Running sequential generation (%d rows)", len(df))
        return await generate_sequential(df, prompt_builder, handler)
    else:
        logger.info("Running concurrent generation (%d rows, max_concurrency=%d)", len(df), max_concurrency)
        return await generate_concurrent(df, prompt_builder, handler, max_concurrency)

class TestRunner:
    """
    Runner for executing model tests on datasets.

    This class loads a dataset (local or Hugging Face), builds prompts
    from each row using a `prompt_builder` function (can be user-provided),
    and generates model responses via `PromptHandler`. Results are saved
    to CSV or JSONL files with timestamps.

    Parameters:
        test_name (str): Name of the test (used in output folder naming).
        test_dataset_path (str): Path to dataset (local or hf:// path).
        model_provider (str): Provider name (e.g., "openai", "anthropic").
        model_name (str): Model name registered in ModelRegistry.
        prompt_builder (Callable): Function that builds a prompt from a dataset row.
        system_prompt (str, optional): Optional system-level instruction.
        dataset_splits (str, optional): Split name for Hugging Face datasets.
        optional_evaluation_output_path (str, optional): Custom path for saving results.
        max_concurrency (int, optional): Maximum number of concurrent requests.
            - 0 → sequential execution (default)
            - >0 → concurrent execution with a semaphore limit

    Notes:
        - Results are saved under `test_results/<test_name>/results_<timestamp>.csv`
          unless a custom output path is provided.
        - Dataset rows can be filtered with `selection` (single index, range, or random).
        - `run_model()` is asynchronous and should be awaited.
    """

    def __init__(
        self,
        test_name: str,
        test_dataset_path: str,
        model_provider: str,
        model_name: str,
        prompt_builder: callable,
        system_prompt: str | list[str] = "",
        dataset_splits: list[str] | None = None,
        evaluation_output_path: str | None = None,
        output_format: str = "jsonl",
        max_concurrency: int = 10
    ):
        self.test_name = test_name
        self.dataset_path = test_dataset_path
        self.model_provider = model_provider
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.dataset_splits = dataset_splits
        self.output_path = evaluation_output_path
        self.output_format = output_format.lower()
        self.max_concurrency = max_concurrency

        self.prompt_builder = prompt_builder

        self.loader = DatasetLoader()
        self.df = None

        self.handler = PromptHandler(
            model_name=self.model_name,
            provider=self.model_provider,
            system_prompt=self.system_prompt
        )

    def load_dataset(self, selection: str = None):
        """Load dataset into memory and apply optional row selection."""
        logger.info("Loading dataset: %s", self.dataset_path)

        if self.dataset_splits:
            df = self.loader.load(self.dataset_path, split=self.dataset_splits)
        else:
            df = self.loader.load(self.dataset_path)

        if selection:
            df = self._select_rows(df, selection)

        self.df = df
        logger.info("Loaded dataset with %d rows", len(df))

    def _select_rows(self, df, selection: str):
        """Helper to filter rows by index, range, or random selection."""
        if selection.isdigit():
            return df.iloc[[int(selection) - 1]]

        if selection.startswith("random:"):
            n = int(selection.split(":")[1])
            return df.sample(n)

        if "-" in selection:
            start, end = selection.split("-")
            start_idx = None if start == "start" else int(start) - 1
            end_idx = None if end == "end" else int(end)
            return df.iloc[start_idx:end_idx]

        raise ValueError(f"Invalid selection format: {selection}")

    def save_results(self, df):
        if self.output_path:
            output_file = self.output_path
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            folder = os.path.join("test_results", self.test_name)
            os.makedirs(folder, exist_ok=True)
            ext = "csv" if self.output_format == "csv" else "jsonl"
            output_file = os.path.join(folder, f"results_{timestamp}.{ext}")

        if self.output_format == "csv":
            df.to_csv(output_file, index=False)
        elif self.output_format == "jsonl":
            df.to_json(output_file, orient="records", lines=True)
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")

        logger.info("Results saved to: %s", output_file)
        return output_file

    async def run_model(self, selection: str = None):
        """
        Run the model on the dataset and save results.

        Parameters:
        selection (str, optional): Row selection format:
            - "10" → single row by index
            - "random:N" → N random rows
            - "start-20" → rows from start to index 20
            - "5-end" → rows from index 5 to end

        Returns:
            str: Path to saved CSV file with model answers.
        """
        logger.info("Running test '%s' using provider=%s model=%s", self.test_name, self.model_provider, self.model_name)
        
        self.load_dataset(selection=selection)
        df = self.df.copy()

        answers = await generate_answers(df, self.prompt_builder, self.handler, self.max_concurrency)

        df["model_answer"] = answers
        output_path = self.save_results(df)
        logger.info("Test '%s' completed successfully", self.test_name)
        return output_path
