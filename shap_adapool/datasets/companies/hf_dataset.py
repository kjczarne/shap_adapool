import pandas as pd
from datasets import Dataset

from .get_data import get_data
from ..common import hf_dataset_from_pandas


def create_hf_dataset(df: pd.DataFrame) -> Dataset:
    return hf_dataset_from_pandas(df)


def main():
    df = get_data()
    dataset = create_hf_dataset(df)
    print(dataset)


if __name__ == "__main__":
    main()
