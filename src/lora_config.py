def apply_lora(model):
    try:
        from unsloth import FastLanguageModel
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "Unsloth could not be imported. Install a PyTorch/Unsloth-compatible "
            "environment before starting fine-tuning."
        ) from error

    model = FastLanguageModel.get_peft_model(
        model,
        r = 8,
        lora_alpha = 16,
        lora_dropout = 0.0,
        bias = "none",

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],

        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    return model