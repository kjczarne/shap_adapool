import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

from .get_data import get_raw_data
from .data_source import DATASET_PATH, CLEAN_DATASET_PATH


def clean_df(df: pd.DataFrame,
            #  revenue_bins: int | Tuple[int, ...] = (0, 10, 100, 1000),
             revenue_bins: int | Tuple[int, ...] = (-1, 1000, np.inf),
             bin_labels: Tuple[str, str] = ("Low", "High"),
             max_revenue: None | float = None,
             years: Tuple[int, ...] = tuple(range(2009, 2019))) -> pd.DataFrame:
    df_ = df.copy(deep=True)
    # if the maximum revenue is unspecified, use the top revenue across all years for all samples
    max_revenue_ = max_revenue
    if max_revenue is None:
        for year in years:
            max_revenue_ = df_[f"Revenue{year}"].max()

    # change all NaNs to average across the selected years, drop rows that do not
    # contain any values within the selected range of time
    for idx, row in df.iterrows():
        rev_from_rows_with_at_least_one_rev = [row[f"Revenue{year}"] for year in years if not np.isnan(row[f"Revenue{year}"])]
        if len(rev_from_rows_with_at_least_one_rev) > 0:
            mean_val = np.mean(rev_from_rows_with_at_least_one_rev)
        else:
            mean_val = np.NaN
        if np.isnan(mean_val):
            df_.drop(idx, inplace=True)
        else:
            for year in years:
                if np.isnan(df_.loc[idx, f"Revenue{year}"]):
                    df_.loc[idx, f"Revenue{year}"] = mean_val

    # create categorical columns for each `RevenueYYYY` column that bin
    # the values into a number of bins within the [0, max] range:
    for year in years:
        df_[f"Revenue{year}C"], bins = pd.cut(df_[f"Revenue{year}"],
                                              bins=revenue_bins,
                                              labels=bin_labels,
                                              retbins=True)
        print(bins)

    # Create a lookup table to convert string labels to integer labels:
    label_dict = {k: v for v, k in enumerate(bin_labels)}

    # Base the label on max occurences of low/high. This is a silly heuristic
    # but should be sufficient to generate XAI samples:
    for idx, row in df_.iterrows():
        values, counts = np.unique([row[f"Revenue{year}C"] for year in years], return_counts=True)
        df_.loc[idx, "label"] = label_dict[values[np.argmax(counts)]]

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

