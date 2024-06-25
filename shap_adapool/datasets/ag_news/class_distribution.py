# %%
import pandas as pd
from pathlib import Path
import argparse

# %%
from .get_data import get_data
from ...initializer import init
# %%


def make_histogram(df: pd.DataFrame, figsize=(40, 5)):
    hist = df["label"].hist(figsize=figsize)
    return hist

# %%

def main():
    parser = argparse.ArgumentParser(description="Creates a histogram for the AG News dataset.")
    args = parser.parse_args()

    init()

    df_train, df_test = get_data()
    hist_train = make_histogram(df_train)
    hist_test = make_histogram(df_test)
    hist_train.get_figure()\
              .savefig("results/ag_news_class_histogram_train.png")
    hist_test.get_figure()\
              .savefig("results/ag_news_class_histogram_test.png")


if __name__ == "__main__":
    main()
