from pathlib import Path
from datasets import Dataset, DatasetDict
from rich.console import Console


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

def save_split(dataset_dict: DatasetDict, path: Path) -> None:
    """Save the split datasets to disk"""
    dataset_dict.save_to_disk(path)


def load_split(path: Path) -> DatasetDict:
    """Load the split datasets from disk"""
    return DatasetDict.load_from_disk(path)
