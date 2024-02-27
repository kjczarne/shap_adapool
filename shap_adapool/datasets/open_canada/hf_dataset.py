import pandas as pd
from datasets import Dataset
from toolz import pipe, compose_left
from .get_data import get_data


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


def naics_sector_to_numerical_id(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the NAICS Sector ID to a numerical ID"""
    # copy is needed to suppress `SettingWithCopyWarning`
    df_ = df.copy(deep=True)
    df_["NAICS Sector EN"] = pd.Categorical(df_["NAICS Sector EN"])
    df_["label"] = df_["NAICS Sector EN"].cat.codes
    return df_


def remove_naics_categorical_column(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the NAICS Sector EN column, as it is no longer needed after
    encoding it as categorical integers"""
    return df.drop(columns=["NAICS Sector EN"])


def rename_text_input_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the text input column to \"text\""""
    return df.rename(columns={"Description (English)": "text"})


def create_hf_dataset(df: pd.DataFrame) -> Dataset:
    """Applies a pipeline of transformations necessary for obtaining a
    Hugging Face dataset from a pandas DataFrame for the
    Open Canada dataset."""
    df = compose_left(
              reduce_columns,
              naics_sector_to_numerical_id,
              remove_naics_categorical_column,
              rename_text_input_column)(df)
    return hf_dataset_from_pandas(df)


def main():
    df = get_data()
    dataset = create_hf_dataset(df)
    print(dataset)


if __name__ == "__main__":
    main()
