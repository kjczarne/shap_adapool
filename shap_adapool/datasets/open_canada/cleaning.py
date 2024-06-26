import argparse
import pandas as pd
from pathlib import Path
from .get_data import get_raw_data
from .data_source import DATASET_PATHS, CLEAN_DATASET_PATH


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # filter only for the top 3 classes which represent the majority of the data and are more balanced
    # than the rest of the classes:
    top_classes = [
        "Manufacturing",
        "Professional, scientific and technical services",
        "Information and cultural industries",
    ]

    df_ = df[df["NAICS Sector EN"].isin(top_classes)]
    
    # get rid of all the rows that contain descriptions that are empty or
    # those that contain multiples of `###`:
    df_ = df_[[not i for i in df_["Description (English)"].str.contains("###")]]
    df_ = df_[df_["Description (English)"].notna()]
    # df_ = df_.where(pd.Series([i != "" for i in df_["Description (English)"].str]))

    return df_


def main():
    parser = argparse.ArgumentParser(description="Cleans the Open Canada dataset.")
    parser.add_argument("--paths",
                        type=str,
                        nargs="+",
                        help="Paths to the Open Canada dataset CSV files.",
                        default=DATASET_PATHS)
    parser.add_argument("--output",
                        type=str,
                        help="Path to the output CSV file.",
                        default=CLEAN_DATASET_PATH)
    args = parser.parse_args()

    paths = tuple(Path(p) for p in args.paths)
    df = get_raw_data(paths)
    df_ = clean_df(df)
    df_.to_csv(Path(args.output), index=False)


if __name__ == "__main__":
    main()
