import pandas as pd
from pathlib import Path

def get_data(path: Path = Path("2023_24_grants_and_contributions.csv")) -> pd.DataFrame:
    """Loads the Open Canada dataset from a CSV file."""
    return pd.read_csv(path, header=0)
