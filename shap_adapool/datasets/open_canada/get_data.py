import pandas as pd
from pathlib import Path
from typing import Tuple

from .data_source import CLEAN_DATASET_PATH

def get_data(path: Path = CLEAN_DATASET_PATH) -> pd.DataFrame:
    """Loads the Open Canada dataset from a CSV file."""
    return pd.read_csv(path, header=0)


def get_raw_data(paths: Tuple[Path]) -> pd.DataFrame:
    """Loads the Open Canada dataset from CSV files and concatenates them into a single DataFrame."""
    return pd.concat([pd.read_csv(path, header=0, encoding="ISO-8859-1") for path in paths], ignore_index=True)
