import argparse
import pandas as pd
from pathlib import Path
from .get_data import get_data


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
    parser = argparse.ArgumentParser(description="Creates a histogram for the Open Canada dataset.")
    parser.add_argument("--path",
                        type=str,
                        help="Path to the Open Canada dataset CSV file.",
                        default="2023_24_grants_and_contributions.csv")
    parser.add_argument("--output",
                        type=str,
                        help="Path to the output CSV file.",
                        default="results/open_canada.csv")
    args = parser.parse_args()

    path = Path(args.path)
    df = get_data(path)
    df_ = clean_df(df)
    df_.to_csv(Path(args.output), index=False)


if __name__ == "__main__":
    main()
