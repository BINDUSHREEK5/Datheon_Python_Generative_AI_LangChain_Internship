
"""Pipeline for prompting, parsing, and validating extracted data."""

import json
import os

from dotenv import load_dotenv
from google import genai
from pydantic import ValidationError

from prompt_templates import (
    INVOICE_EXTRACTION_PROMPT,
    RESUME_EXTRACTION_PROMPT,
)
from schemas import Invoice, Resume


load_dotenv()

MAX_INPUT_LENGTH = 50_000

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

client = genai.Client(api_key=api_key) if api_key else None


def _extract(text: str, extraction_type: str):
    """Call Gemini, parse its JSON response, and validate the data."""

    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError(
            f"Input is too long. Maximum length is {MAX_INPUT_LENGTH} characters."
        )

    if client is None:
        raise RuntimeError("GEMINI_API_KEY is missing. Check your .env file.")

    configurations = {
        "invoice": (INVOICE_EXTRACTION_PROMPT, Invoice, "invoice_text"),
        "resume": (RESUME_EXTRACTION_PROMPT, Resume, "resume_text"),
    }

    if extraction_type not in configurations:
        raise ValueError("Unsupported extraction type. Use 'invoice' or 'resume'.")

    template, schema, placeholder = configurations[extraction_type]
    prompt = template.replace("{" + placeholder + "}", text)

    print(f"Sending {extraction_type} to Gemini... Please wait.")

    try:
        interaction = client.interactions.create(
            model=model_name,
            input=prompt,
            timeout=60,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Gemini API request failed: {type(exc).__name__}."
        ) from exc

    output = interaction.output_text
    if not output or not output.strip():
        raise ValueError("Gemini returned an empty response.")

    # Handle responses wrapped in Markdown code fences.
    output = output.strip()
    if output.startswith("```"):
        lines = output.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        output = "\n".join(lines).strip()
        if output.lower().startswith("json"):
            output = output[4:].strip()

    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        preview = output[:300]
        raise ValueError(
            f"Gemini returned malformed JSON. Response preview: {preview!r}"
        ) from exc

    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            f"Gemini returned invalid {extraction_type} data: {exc}"
        ) from exc


def extract_invoice(invoice_text: str) -> Invoice:
    """Extract invoice details and validate them with Pydantic."""
    return _extract(invoice_text, "invoice")


def extract_resume(resume_text: str) -> Resume:
    """Extract resume details and validate them with Pydantic."""
    return _extract(resume_text, "resume")


def invoice_to_json(invoice: Invoice) -> str:
    """Convert a validated invoice into JSON."""
    return invoice.model_dump_json(indent=2, exclude_none=False)


def resume_to_json(resume: Resume) -> str:
    """Convert a validated resume into JSON."""
    return resume.model_dump_json(indent=2, exclude_none=False)
