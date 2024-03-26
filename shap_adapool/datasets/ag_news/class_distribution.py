# %%
import pandas as pd
from pathlib import Path
import argparse

# %%
from .get_data import get_data
from ...initializer import init
# %%


def make_histogram(df: pd.DataFrame, figsize=(80, 5)):
    hist = df["label"].hist(figsize=figsize)
    return hist

# %%

def main():
    parser = argparse.ArgumentParser(description="Creates a histogram for the AG News dataset.")
    args = parser.parse_args()

    init()

    df = get_data()
    hist = make_histogram(df)
    hist.get_figure()\
        .savefig("results/ag_news_class_histogram.png")


if __name__ == "__main__":
    main()
