import torch
from trl import SFTTrainer
from transformers import TrainingArguments

from .output_config import OUTPUT_DIR


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
):
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            logging_steps=10,
            save_strategy="steps",
            save_steps=250,
            save_total_limit=2,
            eval_strategy="steps",
            eval_steps=250,
            optim="adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            warmup_steps=10,
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            report_to="none",
        ),
    )

    return trainer