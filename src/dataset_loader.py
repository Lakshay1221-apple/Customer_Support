from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


def load_formatted_dataset(
    train_path: str | Path = SPLITS_DIR / "train.jsonl",
    validation_path: str | Path = SPLITS_DIR / "validation.jsonl",
):
    """Load the prepared conversational train and validation splits."""
    train_path = Path(train_path)
    validation_path = Path(validation_path)
    if not train_path.is_absolute():
        train_path = PROJECT_ROOT / train_path
    if not validation_path.is_absolute():
        validation_path = PROJECT_ROOT / validation_path

    missing = [path for path in (train_path, validation_path) if not path.exists()]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Prepared dataset split(s) not found: {paths}")

    train_dataset = load_dataset("json", data_files=str(train_path), split="train")
    validation_dataset = load_dataset(
        "json", data_files=str(validation_path), split="train"
    )
    return train_dataset, validation_dataset