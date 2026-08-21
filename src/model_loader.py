MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True


def load_model_and_tokenizer():
    try:
        from unsloth import FastLanguageModel
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "Unsloth could not be imported. Install a PyTorch/Unsloth-compatible "
            "environment before starting fine-tuning."
        ) from error

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        load_in_4bit=LOAD_IN_4BIT,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    return model, tokenizer