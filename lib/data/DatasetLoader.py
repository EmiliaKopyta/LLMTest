import os
import pandas as pd
from typing import Optional

class DatasetLoader:
    """
    Utility class for loading datasets into pandas DataFrames.

    Supports:
    - Local files (.csv, .json, .jsonl, .parquet)
    - Hugging Face-hosted files via `hf://` paths
    - Recursive folder loading of multiple supported files
    """

    LOADERS = {
        ".csv": lambda fp: pd.read_csv(fp),
        ".json": lambda fp: pd.read_json(fp),
        ".jsonl": lambda fp: pd.read_json(fp, lines=True),
        ".parquet": lambda fp: pd.read_parquet(fp),
    }

    @staticmethod
    def load(path: str, split: Optional[str] = None) -> pd.DataFrame:
        """
        Load a dataset from a local file or Hugging Face-hosted file.

        Parameters:
            path (str): Either a local path or an hf:// path.
            split (str, optional): For Hugging Face datasets requiring split info.

        Returns:
            pd.DataFrame: Loaded dataset.

        Notes:
            - For Hugging Face datasets, it is generally recommended to use the
            "Use this dataset -> pandas" integration provided on the dataset page,
            as it ensures the most efficient and reliable loading.
            - The `load()` method cannot copy that integration directly, but extracts the
            necessary information (file format, split, path).
             - Folder loading will combine all supported files into a single DataFrame.
            If files have different schemas, missing columns will be filled with NaN.
        """
        if path.startswith("hf://"):
            return DatasetLoader._load_huggingface(path, split)
        elif os.path.isdir(path):
            return DatasetLoader._load_from_folder(path)
        else:
            return DatasetLoader._load_local_file(path)

    @staticmethod
    def _load_huggingface(path: str, split: Optional[str]) -> pd.DataFrame:
        """Load dataset from Hugging Face using pandas."""
        base_path = path.replace("hf://", "")
        full_path = os.path.join(base_path, split) if split else base_path

        ext = next((e for e in DatasetLoader.LOADERS if full_path.endswith(e)), None)
        if not ext:
            raise ValueError(
                f"Unsupported Hugging Face file format: {full_path}. "
                f"Supported formats: {', '.join(DatasetLoader.LOADERS)}"
            )

        return DatasetLoader.LOADERS[ext]("hf://" + full_path)

    @staticmethod
    def _load_local_file(path: str) -> pd.DataFrame:
        """Load a single local dataset file using pandas."""
        ext = next((e for e in DatasetLoader.LOADERS if path.endswith(e)), None)
        if not ext:
            raise ValueError(
                f"Unsupported file format: {path}. "
                f"Supported formats: {', '.join(DatasetLoader.LOADERS)}"
            )

        return DatasetLoader.LOADERS[ext](path)

    @staticmethod
    def _load_from_folder(folder_path: str) -> pd.DataFrame:
        """Load and combine supported dataset files from a local folder recursively."""
        files = [
            os.path.join(root, file)
            for root, _, file_list in os.walk(folder_path)
            for file in file_list
            if any(file.endswith(ext) for ext in DatasetLoader.LOADERS)
        ]

        if not files:
            raise FileNotFoundError(
                f"No supported files ({', '.join(DatasetLoader.LOADERS)}) found in folder: {folder_path}"
            )

        dfs = []
        for fp in files:
            ext = next((e for e in DatasetLoader.LOADERS if fp.endswith(e)), None)
            dfs.append(DatasetLoader.LOADERS[ext](fp))

        return pd.concat(dfs, ignore_index=True)
