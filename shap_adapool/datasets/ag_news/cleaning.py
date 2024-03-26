import argparse
import pandas as pd
from pathlib import Path
from .get_data import get_data
from .data_source import DATASET_PATH, CLEAN_DATASET_PATH


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df


def main():
    parser = argparse.ArgumentParser(description="Creates a histogram for the Open Canada dataset.")
    parser.add_argument("--output",
                        type=str,
                        help="Path to the output CSV file.",
                        default=CLEAN_DATASET_PATH)
    args = parser.parse_args()

    path = Path(args.path)
    df = get_data(path)
    df_ = clean_df(df)
    df_.to_csv(Path(args.output), index=False)


if __name__ == "__main__":
    main()
