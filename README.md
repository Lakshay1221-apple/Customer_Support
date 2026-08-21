# Customer Support AI

A production-oriented fine-tuning pipeline for building a customer support assistant from multi-language support ticket data. The project combines data preparation, validation, and parameter-efficient LLM training to convert raw ticket data into a high-quality instruction-tuning dataset and a locally trained model suitable for support workflows.

## Overview

This repository is designed to:

- clean and validate raw support ticket data,
- convert tickets into classification and instruction-finetuning examples,
- split data into train/validation/test sets,
- generate a quality report for dataset health,
- fine-tune a model using LoRA/QLoRA-style adaptation with Unsloth and Transformers,
- save a final model artifact ready for local inference or deployment.

The project is built around a customer support ticket dataset and uses the `unsloth/Llama-3.2-3B-Instruct` base model for efficient fine-tuning.

## Key Features

- End-to-end data pipeline from raw CSV to curated training-ready JSONL files
- Conservative text cleaning and normalization for multilingual ticket content
- Deduplication and filtering to reduce noise and low-quality examples
- Structured dataset conversion for:
  - classification tasks,
  - supervised fine-tuning (SFT) conversations,
  - train/validation/test splits
- Automatic dataset reporting and token statistics
- LoRA-based fine-tuning configuration for efficient adaptation
- GPU-oriented setup for local training workflows

## Repository Structure

```text
Customer_Support/
├── data/
│   ├── raw/
│   │   └── dataset-tickets-multi-lang-4-20k.csv
│   ├── processed/
│   │   ├── cleaned.csv
│   │   ├── filtered.csv
│   │   ├── deduplicated.csv
│   │   ├── classification_dataset.jsonl
│   │   └── sft_dataset.jsonl
│   ├── splits/
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   └── dataset-tickets-multi-lang-4-20k.csv
├── reports/
│   └── data_quality_report.json
├── src/
│   ├── data/
│   │   ├── __main__.py
│   │   ├── analyze.py
│   │   ├── clean.py
│   │   ├── config.py
│   │   ├── deduplicate.py
│   │   ├── filter.py
│   │   ├── format.py
│   │   ├── load.py
│   │   ├── pipeline.py
│   │   ├── split.py
│   │   └── validate.py
│   ├── dataset_loader.py
│   ├── generation_config.py
│   ├── lora_config.py
│   ├── model_loader.py
│   ├── output_config.py
│   ├── train.py
│   ├── trainer.py
│   └── __init__.py
├── tests/
│   └── test_data_pipeline.py
├── .gitignore
├── pyproject.toml
├── README.md
├── requirements.txt
└── outputs/
    └── final_model/
```

## Data Pipeline

The data workflow is orchestrated through the pipeline under `src/data/pipeline.py`.

It performs the following steps:

1. Load the raw CSV dataset
2. Profile the dataset for quality insights
3. Select required columns
4. Clean and normalize text values
5. Filter out invalid or irrelevant records
6. Remove near-duplicate support entries
7. Convert rows into classification and SFT examples
8. Validate instruction-format integrity
9. Split data into train/validation/test sets
10. Save processed outputs and generate a data quality report

A sample report is generated in `reports/data_quality_report.json` and includes:

- example counts,
- removal reasons,
- label distribution,
- token statistics,
- split sizes.

## Training Workflow

The training entry point is `src/train.py`, which:

- loads the prepared train and validation datasets,
- loads the base model and tokenizer,
- applies LoRA adaptation,
- creates a Hugging Face `SFTTrainer`,
- trains the model,
- saves the final model into `outputs/final_model`.

### Current Training Configuration

The default setup uses:

- Base model: `unsloth/Llama-3.2-3B-Instruct`
- LoRA rank: `8`
- LoRA alpha: `16`
- Max sequence length: `2048`
- Training epochs: `3`
- Learning rate: `2e-4`
- Optimizer: `adamw_8bit` when CUDA is available, otherwise `adamw_torch`

## Prerequisites

Before running this project, ensure the following are available:

- Python 3.12+
- NVIDIA GPU recommended for fine-tuning
- CUDA-enabled PyTorch environment
- At least 16 GB RAM recommended for local experimentation
- Internet access to download model weights and packages

## Installation

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

If you are using a CUDA 12.1 environment, the project is already configured for the PyTorch CUDA wheel in `requirements.txt`.

## Quick Start

### 1. Prepare the dataset

Place the raw CSV in the repository at:

```text
data/raw/dataset-tickets-multi-lang-4-20k.csv
```

Then run:

```bash
python -m src.data
```

This generates processed datasets and reports in:

```text
data/processed/
data/splits/
reports/
```

### 2. Train the model

```bash
python -m src.train
```

The trained model artifact is saved to:

```text
outputs/final_model/
```

## Output Artifacts

After the pipeline runs, the project creates the following high-value outputs:

- `data/processed/cleaned.csv`
- `data/processed/filtered.csv`
- `data/processed/deduplicated.csv`
- `data/processed/classification_dataset.jsonl`
- `data/processed/sft_dataset.jsonl`
- `data/splits/train.jsonl`
- `data/splits/validation.jsonl`
- `data/splits/test.jsonl`
- `reports/data_quality_report.json`
- `outputs/final_model/`

## Model and Training Notes

This project uses parameter-efficient fine-tuning instead of full-model retraining. That keeps the workflow lighter and more practical for local development while still providing strong task adaptation for customer support scenarios.

The pipeline is intentionally structured to be reproducible and inspectable, which makes it useful for experimentation, evaluation, and iterative tuning.

## Testing

The project includes a basic validation suite for the data transformation layer:

```bash
pytest
```

Current tests verify:

- conservative text cleaning,
- deduplication behavior,
- SFT record formatting and JSON validation.

## Important Considerations

- The project expects a valid support ticket CSV as input.
- The base model download and fine-tuning are GPU-intensive operations.
- If CUDA or Unsloth dependencies are missing, the trainer will fail with a clear runtime error.
- Carefully review `reports/data_quality_report.json` to understand dataset quality before training.

## Future Enhancements

Potential next steps include:

- adding evaluation metrics for response quality,
- integrating a local inference script,
- adding support for multilingual response generation evaluation,
- exposing training configuration through command-line flags,
- packaging the project for easier deployment and experimentation.

## License

This project is intended for research and internal experimentation. Add your preferred open-source license if you plan to distribute it publicly.

## Author

Lakshay Raj
