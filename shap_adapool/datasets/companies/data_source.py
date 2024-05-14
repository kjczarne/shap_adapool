from pathlib import Path
import os
import numpy as np


VARIANT = os.environ.get("VARIANT", "public")

match VARIANT:
    case "public":
        DATASET_PATH = Path("Public Large Firms Example.csv")
        CLEAN_DATASET_PATH = Path("results/public.csv")
        DATASET_OUTPUT_PATH = Path("results/public")
        BIN_RANGE = (-1, 100, np.inf)
        YEARS_RANGE = tuple(range(2019, 2021))
    case "private":
        DATASET_PATH = Path(r"D:\Dataset\Tech-2020-02-12.csv")
        CLEAN_DATASET_PATH = Path(r"D:\shap_adapool_results\private.csv")
        DATASET_OUTPUT_PATH = Path(r"D:\shap_adapool_results\private")
        BIN_RANGE = (-1, 1000, np.inf)
        YEARS_RANGE = tuple(range(2009, 2019))
    case _:
        raise ValueError(f"Unknown variant: {VARIANT}")
