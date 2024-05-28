from datasets import load_dataset, Dataset
from .data_source import DATASET_NAME


def create_hf_dataset() -> Dataset:
    dataset = load_dataset(DATASET_NAME)
    return dataset
    

def main():
    dataset = create_hf_dataset()
    print(dataset)


if __name__ == "__main__":
    main()
