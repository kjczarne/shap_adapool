import argparse
import pandas as pd
from pathlib import Path
from typing import Tuple

from .get_data import get_raw_data
from .data_source import DATASET_PATH, CLEAN_DATASET_PATH


def clean_df(df: pd.DataFrame,
            #  revenue_bins: int | Tuple[int, ...] = (0, 10, 100, 1000),
             revenue_bins: int | Tuple[int, ...] = (0, 99, 1000),
             bin_labels: Tuple[str, str] = ("Low", "High"),
             max_revenue: None | float = None,
             years: Tuple[int, ...] = (2019, 2020, 2021, 2022)) -> pd.DataFrame:
    df_ = df.copy(deep=True)
    # if the maximum revenue is unspecified, use the top revenue across all years for all samples
    max_revenue_ = max_revenue
    if max_revenue is None:
        for year in years:
            max_revenue_ = df_[f"Revenue{year}"].max()
    # create categorical columns for each `RevenueYYYY` column that bin
    # the values into a number of bins within the [0, max] range:
    for year in years:
        df_[f"Revenue{year}C"], bins = pd.cut(df_[f"Revenue{year}"], bins=revenue_bins, labels=bin_labels, retbins=True)
        print(bins)
    return df_


def main():
    parser = argparse.ArgumentParser(description="Cleans the Companies dataset.")
    parser.add_argument("--path",
                        type=str,
                        nargs="+",
                        help="Paths to the companies dataset CSV file.",
                        default=DATASET_PATH)
    parser.add_argument("--output",
                        type=str,
                        help="Path to the output CSV file.",
                        default=CLEAN_DATASET_PATH)
    args = parser.parse_args()

    df = get_raw_data(args.path)
    df_ = clean_df(df)
    df_.to_csv(Path(args.output), index=False)
    print(df_.head())


if __name__ == "__main__":
    main()

