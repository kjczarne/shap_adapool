# %%
import pandas as pd
from pathlib import Path
import argparse

# %%
from ...initializer import init
# %%


def make_histogram(path: Path, figsize=(80, 5)):
    
    df = pd.read_csv(path, header=0)
    hist = df["NAICS Sector EN"].hist(figsize=figsize)
    return hist

# %%

def main():
    parser = argparse.ArgumentParser(description="Creates a histogram for the Open Canada dataset.")
    parser.add_argument("--path",
                        type=str,
                        help="Path to the Open Canada dataset CSV file.",
                        default="2023_24_grants_and_contributions.csv")
    args = parser.parse_args()

    init()

    path = Path(args.path)
    hist = make_histogram(path)
    hist.get_figure()\
        .savefig("results/naics_sector_en_histogram.png")


if __name__ == "__main__":
    main()
