import pandas as pd
from pathlib import Path

from .data_source import CLEAN_DATASET_PATH, DATASET_PATH

def get_data(path: Path = CLEAN_DATASET_PATH) -> pd.DataFrame:
    """Loads the Companies dataset from a CSV file."""
    return pd.read_csv(path, header=0)


def get_raw_data(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Loads the Companies dataset from a CSV file."""
    return pd.read_csv(path, header=0, encoding="ISO-8859-1")
