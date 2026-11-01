
import json
from pydantic import ValidationError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import extractor
from schemas import Invoice, Resume


@pytest.fixture
def fake_client(monkeypatch):
    client = Mock()
    monkeypatch.setattr(extractor, "client", client)
    return client


def set_response(client, data):
    client.interactions.create.return_value = SimpleNamespace(
        output_text=json.dumps(data)
    )


def test_invoice_extraction(fake_client):
    set_response(fake_client, {
        "invoice_number": "INV-104",
        "vendor": "ACME Technologies",
        "invoice_date": "2026-10-12",
        "due_date": "2026-10-30",
        "total_amount": "2500",
        "currency": "INR",
        "line_items": [],
    })

    result = extractor.extract_invoice("Sample invoice text")

    assert isinstance(result, Invoice)
    assert result.invoice_number == "INV-104"
    assert str(result.total_amount) == "2500"
    fake_client.interactions.create.assert_called_once()


def test_resume_extraction(fake_client):
    set_response(fake_client, {
        "candidate_name": "Jane Doe",
        "email": "jane.doe@example.com",
        "phone": None,
        "skills": ["Python", "SQL"],
        "education": [],
        "work_experience": [],
        "projects": [],
    })

    result = extractor.extract_resume("Sample resume text")

    assert isinstance(result, Resume)
    assert result.candidate_name == "Jane Doe"
    assert result.skills == ["Python", "SQL"]


def test_malformed_json(fake_client):
    fake_client.interactions.create.return_value = SimpleNamespace(
        output_text="this is not JSON"
    )

    with pytest.raises(ValueError, match="malformed JSON"):
        extractor.extract_invoice("Sample invoice text")


def test_invalid_invoice_data(fake_client):
    set_response(fake_client, {
        "total_amount": "-50",
        "currency": "INR",
    })

    with pytest.raises(ValueError, match="invalid invoice data"):
        extractor.extract_invoice("Sample invoice text")


@pytest.mark.parametrize("function", [
    extractor.extract_invoice,
    extractor.extract_resume,
])
def test_empty_input_does_not_call_api(fake_client, function):
    with pytest.raises(ValueError, match="cannot be empty"):
        function("  ")

    fake_client.interactions.create.assert_not_called()


def test_missing_api_key(monkeypatch):
    monkeypatch.setattr(extractor, "client", None)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is missing"):
        extractor.extract_invoice("Sample invoice text")
def test_invoice_rejects_missing_required_field():
    data = {
        "vendor": "ACME Technologies",
        "invoice_date": "2026-10-12",
        "due_date": "2026-10-30",
        "total_amount": "2500",
        "currency": "INR",
        "line_items": [],
    }

    with pytest.raises(ValidationError):
        Invoice.model_validate(data)


def test_resume_rejects_invalid_email():
    data = {
        "candidate_name": "Jane Doe",
        "email": "not-an-email",
        "phone": None,
        "skills": ["Python"],
        "education": [],
        "work_experience": [],
        "projects": [],
    }

    with pytest.raises(ValidationError):
        Resume.model_validate(data)


def test_invoice_rejects_unexpected_field():
    data = {
        "invoice_number": "INV-104",
        "vendor": "ACME Technologies",
        "invoice_date": "2026-10-12",
        "due_date": "2026-10-30",
        "total_amount": "2500",
        "currency": "INR",
        "line_items": [],
        "unexpected_field": "should be rejected",
    }

    with pytest.raises(ValidationError):
        Invoice.model_validate(data)


def test_invoice_rejects_due_date_before_invoice_date():
    data = {
        "invoice_number": "INV-104",
        "vendor": "ACME Technologies",
        "invoice_date": "2026-10-30",
        "due_date": "2026-10-12",
        "total_amount": "2500",
        "currency": "INR",
        "line_items": [],
    }

    with pytest.raises(ValidationError):
        Invoice.model_validate(data)
def test_empty_gemini_response(fake_client):
    fake_client.interactions.create.return_value = SimpleNamespace(
        output_text=""
    )

    with pytest.raises(ValueError, match="empty response"):
        extractor.extract_invoice("Sample invoice text")


def test_gemini_api_failure(fake_client):
    fake_client.interactions.create.side_effect = TimeoutError(
        "Request timed out"
    )

    with pytest.raises(RuntimeError, match="Gemini API request failed"):
        extractor.extract_invoice("Sample invoice text")


def test_oversized_input_does_not_call_api(fake_client):
    text = "a" * (extractor.MAX_INPUT_LENGTH + 1)

    with pytest.raises(ValueError, match="Input is too long"):
        extractor.extract_invoice(text)

    fake_client.interactions.create.assert_not_called()


def test_invoice_extraction_with_markdown_code_fence(fake_client):
    response = {
        "invoice_number": "INV-105",
        "vendor": "Example Ltd",
        "invoice_date": "2026-10-12",
        "due_date": "2026-10-30",
        "total_amount": "500",
        "currency": "INR",
        "line_items": [],
    }

    fake_client.interactions.create.return_value = SimpleNamespace(
        output_text=f"```json\n{json.dumps(response)}\n```"
    )

    result = extractor.extract_invoice("Sample invoice text")

    assert result.invoice_number == "INV-105"