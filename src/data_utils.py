"""Data utility functions for handling and processing customer support ticket data."""

from pathlib import Path

from datasets import load_dataset
from .data.config import RAW_PATH

DATASET_URL = (
    "https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/"
    "resolve/main/dataset-tickets-multi-lang-4-20k.csv"
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = RAW_PATH



def load_customer_support_dataset():
    """Load the multi-language ticket dataset from the specific CSV file.

    The hub repository contains multiple CSV files with different column sets,
    so loading the whole dataset repository without specifying the file causes a
    schema mismatch during dataset generation.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Download it from {DATASET_URL}."
        )

    return load_dataset("csv", data_files=str(DATASET_PATH), split="train")


if __name__ == "__main__":
    dataset = load_customer_support_dataset()
    print(dataset)
    print(dataset.column_names)
    print(dataset[0])