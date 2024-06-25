from pathlib import Path

DATASET_PATHS = (
    Path("2023_24_grants_and_contributions.csv"),
    Path("2022_23_grants_and_contributions.csv"),
    Path("2021_22_grants_and_contributions.csv"),
    Path("2020_21_grants_and_contributions.csv"),
    Path("2019_20_grants_and_contributions.csv"),
    Path("2018_19_grants_and_contributions.csv")
)
CLEAN_DATASET_PATH = Path("results/open_canada.csv")
DATASET_OUTPUT_PATH = Path("results/open_canada_dataset")
