import pandas as pd
from datasets import Dataset, DatasetDict


def hf_dataset_from_pandas(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df)

