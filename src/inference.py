"""Inference module for evaluating customer support ticket classification with fine-tuned models."""

import argparse
import json
from pathlib import Path

from .data.format import INSTRUCTION
from .generation_config import get_generation_config
from .output_config import FINAL_MODEL_DIR



def load_inference_model(model_path: str | Path = FINAL_MODEL_DIR):
    """Load base or fine-tuned model and tokenizer for inference.
    
    Supports Unsloth FastLanguageModel with fallback to Hugging Face Transformers/PEFT.
    """
    model_path = Path(model_path)
    
    try:
        from unsloth import FastLanguageModel
        
        target_path = str(model_path) if model_path.exists() else "unsloth/Llama-3.2-3B-Instruct"
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=target_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return model, tokenizer
    except (ImportError, AttributeError, Exception):
        # Fallback to Hugging Face Transformers & PEFT
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base_model_name = "unsloth/Llama-3.2-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path) if (model_path / "tokenizer_config.json").exists() else base_model_name
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        if model_path.exists() and (model_path / "adapter_config.json").exists():
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path) if model_path.exists() else base_model_name,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
        model.eval()
        return model, tokenizer


def format_prompt(subject: str, body: str) -> str:
    """Format ticket subject and body into system instruction prompt."""
    return f"{INSTRUCTION}\n\nSubject:\n{subject.strip()}\n\nCustomer Message:\n{body.strip()}"


def predict_ticket(
    subject: str,
    body: str,
    model=None,
    tokenizer=None,
    model_path: str | Path = FINAL_MODEL_DIR,
) -> dict:
    """Predict ticket type, queue, and priority from subject and body."""
    if model is None or tokenizer is None:
        model, tokenizer = load_inference_model(model_path)

    prompt = format_prompt(subject, body)
    messages = [{"role": "user", "content": prompt}]
    
    if hasattr(tokenizer, "apply_chat_template"):
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        full_text = f"User: {prompt}\nAssistant:"
        inputs = tokenizer(full_text, return_tensors="pt")

    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_config = get_generation_config()
    
    import torch
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config)

    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_len:]
    decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    try:
        parsed = json.loads(decoded_text)
        return {"success": True, "prediction": parsed, "raw_response": decoded_text}
    except json.JSONDecodeError:
        return {"success": False, "prediction": None, "raw_response": decoded_text, "error": "Invalid JSON response"}


def main():
    parser = argparse.ArgumentParser(description="Run inference on a customer support ticket.")
    parser.add_argument("--subject", type=str, help="Subject of the support ticket")
    parser.add_argument("--body", type=str, help="Body/Content of the support ticket")
    parser.add_argument("--model_path", type=str, default=str(FINAL_MODEL_DIR), help="Path to saved model artifact")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive CLI mode")

    args = parser.parse_args()

    if args.interactive:
        print("=" * 50)
        print("Customer Support Ticket Inference CLI")
        print("=" * 50)
        model, tokenizer = load_inference_model(args.model_path)
        while True:
            try:
                subject = input("\nEnter ticket subject (or 'exit' to quit): ").strip()
                if subject.lower() in ("exit", "quit"):
                    break
                body = input("Enter ticket body: ").strip()
                result = predict_ticket(subject, body, model=model, tokenizer=tokenizer)
                print("\nResult:", json.dumps(result, indent=2))
            except KeyboardInterrupt:
                break
    elif args.subject and args.body:
        result = predict_ticket(args.subject, args.body, model_path=args.model_path)
        print(json.dumps(result, indent=2))
    else:
        sample_subject = "Unable to reset password"
        sample_body = "I have tried clicking forgot password link but haven't received any email reset link for 2 hours."
        print(f"No ticket provided. Running sample inference on:\nSubject: {sample_subject}\nBody: {sample_body}\n")
        result = predict_ticket(sample_subject, sample_body, model_path=args.model_path)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
