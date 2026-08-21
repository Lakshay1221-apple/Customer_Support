import json

from src.data.clean import clean_text
from src.data.deduplicate import deduplicate_rows
from src.data.format import to_sft_rows
from src.data.validate import validate_sft_rows


def test_clean_text_is_conservative():
    assert clean_text(None) == ""
    assert clean_text("  hello \r\n\r\n world  ") == "hello\n\n world"
    assert clean_text("Café https://example.com") == "Café https://example.com"


def test_deduplicate_uses_subject_and_body():
    rows = [{"subject": "A", "body": "B"}, {"subject": " a ", "body": "b", "answer": "different"}]
    unique, removed = deduplicate_rows(rows)
    assert len(unique) == 1
    assert removed == 1


def test_sft_format_has_valid_json_target():
    row = {"subject": "Help", "body": "Need help", "type": "Request", "queue": "Billing", "priority": "low"}
    formatted = to_sft_rows([row])
    assert not validate_sft_rows(formatted)
    assert json.loads(formatted[0]["messages"][1]["content"]) == {"type": "Request", "queue": "Billing", "priority": "low"}
