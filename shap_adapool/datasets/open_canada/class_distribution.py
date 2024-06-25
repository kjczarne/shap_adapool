# %%
import pandas as pd
from pathlib import Path
import argparse

# %%
from .get_data import get_raw_data, get_data
from .data_source import DATASET_PATHS
from ...initializer import init
# %%


def make_histogram(df: pd.DataFrame, figsize=(80, 5)):
    hist = df["NAICS Sector EN"].hist(figsize=figsize)
    return hist

def print_value_counts(df: pd.DataFrame):
    print(df["NAICS Sector EN"].value_counts())

# %%

def main():
    parser = argparse.ArgumentParser(description="Creates a histogram for the Open Canada dataset.")
    parser.add_argument("--paths",
                        type=str,
                        nargs="+",
                        help="Paths to the Open Canada dataset CSV files.",
                        default=DATASET_PATHS)
    parser.add_argument("--post-cleaning",
                        action="store_true",
                        help="Whether to use the cleaned dataset.")
    args = parser.parse_args()

    init()

    paths = tuple(Path(p) for p in args.paths)
    if args.post_cleaning:
        df = get_data()
        out_hist_path = "results/naics_sector_en_histogram_post_cleaning.png"
    else:
        df = get_raw_data(paths)
        out_hist_path = "results/naics_sector_en_histogram_pre_cleaning.png"
    hist = make_histogram(df)
    print_value_counts(df)
    hist.get_figure()\
        .savefig(out_hist_path)


if __name__ == "__main__":
    main()
