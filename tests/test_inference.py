import json
from unittest.mock import MagicMock
from src.generation_config import get_generation_config
from src.inference import format_prompt, predict_ticket
from src.output_config import FINAL_MODEL_DIR, OUTPUT_DIR


def test_generation_config_loaded():
    config = get_generation_config()
    assert config.max_new_tokens == 512
    assert config.temperature == 0.7
    assert config.do_sample is True


def test_format_prompt_incorporates_subject_and_body():
    prompt = format_prompt("Billing Error", "I was charged twice.")
    assert "Subject:\nBilling Error" in prompt
    assert "Customer Message:\nI was charged twice." in prompt
    assert "type, queue, priority" in prompt


def test_output_config_paths():
    assert str(OUTPUT_DIR).endswith("outputs")
    assert str(FINAL_MODEL_DIR).endswith("final_model")


def test_predict_ticket_with_mock_model():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_tokenizer.apply_chat_template.return_value = {"input_ids": MagicMock(shape=(1, 10))}
    mock_tokenizer.decode.return_value = '{"type": "Incident", "queue": "Billing and Payments", "priority": "high"}'
    
    mock_outputs = MagicMock()
    mock_outputs.__getitem__.return_value = MagicMock()
    mock_model.generate.return_value = mock_outputs

    mock_model.device = "cpu"

    result = predict_ticket(
        subject="Payment Failed",
        body="Card was declined",
        model=mock_model,
        tokenizer=mock_tokenizer,
    )

    assert result["success"] is True
    assert result["prediction"]["type"] == "Incident"
    assert result["prediction"]["queue"] == "Billing and Payments"
    assert result["prediction"]["priority"] == "high"
