from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict
from toolz import pipe, compose_left
from functools import partial
from typing import List
from rich.console import Console

from .get_data import get_data

TOP_CLASSES = [
    "Professional, scientific and technical services",
    "Manufacturing",
    "Information and cultural industries",
    "Wholesale trade"
]

DATASET_OUTPATH_DEFAULT = Path("results/hf_dataset")


def hf_dataset_from_pandas(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df)


def reduce_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Take only the description and NAICS Sector before
    feeding this into the language model"""
    return df[["Description (English)", "NAICS Sector EN"]]


def naics_sector_to_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the NAICS Sector ID
    
    > [!warning]
    > Currently unused
    """
    one_hot = pd.get_dummies(df["NAICS Sector EN"])
    return pd.concat([df, one_hot], axis=1)


def naics_sector_to_numerical_id(df: pd.DataFrame,
                                 mapping_save_path: Path = Path("results/naics_labels.csv")) -> pd.DataFrame:
    """Convert the NAICS Sector ID to a numerical ID"""
    # copy is needed to suppress `SettingWithCopyWarning`
    df_ = df.copy(deep=True)
    df_["NAICS Sector EN"] = pd.Categorical(df_["NAICS Sector EN"])
    df_["label"] = df_["NAICS Sector EN"].cat.codes
    # saving a mapping so that we know which label corresponds to which NAICS Sector:
    df_[["NAICS Sector EN", "label"]].drop_duplicates().to_csv(mapping_save_path, index=False)
    return df_


def remove_naics_categorical_column(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the NAICS Sector EN column, as it is no longer needed after
    encoding it as categorical integers"""
    return df.drop(columns=["NAICS Sector EN"])


def rename_text_input_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the text input column to \"text\""""
    return df.rename(columns={"Description (English)": "text"})


def reduce_to_top_classes(df: pd.DataFrame, top_classes: List[str]) -> pd.DataFrame:
    df_ = df[df["NAICS Sector EN"].isin(top_classes)]
    return df_


def create_hf_dataset(df: pd.DataFrame, top_classes: List[str]) -> Dataset:
    """Applies a pipeline of transformations necessary for obtaining a
    Hugging Face dataset from a pandas DataFrame for the
    Open Canada dataset."""
    df = compose_left(
              reduce_columns,
              partial(reduce_to_top_classes, top_classes=top_classes),
              naics_sector_to_numerical_id,
              remove_naics_categorical_column,
              rename_text_input_column)(df)
    return hf_dataset_from_pandas(df)


def train_val_test_split(dataset: Dataset | DatasetDict,
                         test_size: float = 0.1,
                         val_size: float = 0.1) -> DatasetDict:
    """Split the dataset into training, validation, and test sets"""
    console = Console()
    if type(dataset) is DatasetDict:
        if "train" in dataset and "val" in dataset and "test" in dataset:
            console.print("[red]Dataset already split into training, validation, and test sets.[/red]")
            return dataset
        elif "train" in dataset and "test" in dataset and "val" not in dataset:
            console.print("[red]Dataset already split into training and test sets.[/red]")
            console.print("[red]Splitting the test set into test and val[/red]")
            test_and_val = dataset["test"].train_test_split(test_size=(val_size / (test_size + val_size)))
            dataset_ = DatasetDict(train=dataset["train"], test=test_and_val["train"], val=test_and_val["test"])
            return dataset_
        else:
            raise ValueError("DatasetDict must contain at least a training and test set.")
    elif type(dataset) is Dataset:
        train_test_and_val = dataset.train_test_split(test_size=test_size + val_size)
        train = train_test_and_val["train"]
        test_and_val = train_test_and_val["test"].train_test_split(test_size=(val_size / (test_size + val_size)))
        val = test_and_val["train"]
        test = test_and_val["test"]
        return DatasetDict(train=train, val=val, test=test)
    else:
        raise ValueError("Dataset must be of type Dataset or DatasetDict.")


def save_split(dataset_dict: DatasetDict,
               path: Path = DATASET_OUTPATH_DEFAULT) -> None:
    """Save the split datasets to disk"""
    dataset_dict.save_to_disk(path)


def load_split(path: Path = DATASET_OUTPATH_DEFAULT) -> DatasetDict:
    """Load the split datasets from disk"""
    return DatasetDict.load_from_disk(path)


def main():
    df = get_data()
    dataset = create_hf_dataset(df, TOP_CLASSES)
    print(dataset)


if __name__ == "__main__":
    main()
