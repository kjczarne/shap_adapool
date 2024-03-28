import pandas as pd
from pathlib import Path

from .hf_dataset import create_hf_dataset

def get_data() -> (pd.DataFrame, pd.DataFrame):
    """Loads the AG News dataset as a DataFrame"""
    d = create_hf_dataset()
    return d['train'].to_pandas(), d['test'].to_pandas()
