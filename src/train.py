from pathlib import Path

from .dataset_loader import load_formatted_dataset


def main():
    from .lora_config import apply_lora
    from .model_loader import load_model_and_tokenizer

    train_dataset, val_dataset = load_formatted_dataset()

    model, tokenizer = load_model_and_tokenizer()

    model = apply_lora(model)

    from .trainer import create_trainer

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    print("\nTrainer Created Successfully!\n")
    print(trainer)

    trainer.train()

    output_dir = Path("outputs/final_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()