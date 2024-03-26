import pandas as pd
from pathlib import Path

from .data_source import CLEAN_DATASET_PATH
from .hf_dataset import create_hf_dataset

def get_data() -> pd.DataFrame:
    """Loads the AG News dataset as a DataFrame"""
    return create_hf_dataset().to_pandas()
