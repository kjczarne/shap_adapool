import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict
from rich.console import Console


def hf_dataset_from_pandas(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df)


def hf_dataset_to_disk(dataset: Dataset, path: Path) -> None:
    console = Console()
    console.print(f"[red]Saving dataset to {path}[/red]")
    dataset.save_to_disk(str(path))
