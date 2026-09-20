from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FINAL_MODEL_DIR = OUTPUT_DIR / "final_model"

LORA_SAVE_DIR = PROJECT_ROOT / "models" / "lora" / "athena-v1"
MERGED_MODEL_DIR = PROJECT_ROOT / "models" / "merged" / "athena-v1"