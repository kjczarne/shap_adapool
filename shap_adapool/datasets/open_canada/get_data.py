import pandas as pd
from pathlib import Path

from .data_source import CLEAN_DATASET_PATH

def get_data(path: Path = CLEAN_DATASET_PATH) -> pd.DataFrame:
    """Loads the Open Canada dataset from a CSV file."""
    return pd.read_csv(path, header=0)
