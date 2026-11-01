
"""Command-line entry point for the structured document extractor."""

import argparse
import sys
from pathlib import Path

from extractor import (
    extract_invoice,
    extract_resume,
    invoice_to_json,
    resume_to_json,
)


def read_input_file(file_path: str) -> str:
    """Read document text from a UTF-8 text file."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"Input file is empty: {path}")

    return text


def read_interactive_input(document_type: str) -> str:
    """Read pasted text until END is entered on a separate line."""
    print(f"Paste your {document_type} text below.")
    print("When finished, type END on a new line.\n")

    lines = []

    while True:
        try:
            line = input()
        except EOFError:
            break

        if line.strip() == "END":
            break

        lines.append(line)

    text = "\n".join(lines).strip()

    if not text:
        raise ValueError("No document text provided.")

    return text


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured JSON from invoices and resumes."
    )
    parser.add_argument(
        "--type",
        choices=["invoice", "resume"],
        default="invoice",
        help="Document type to extract (default: invoice).",
    )
    parser.add_argument(
        "--file",
        help="Path to a UTF-8 text file. If omitted, paste text interactively.",
    )
    parser.add_argument(
        "--output",
        help="Optional path where the extracted JSON will be saved.",
    )

    args = parser.parse_args()

    try:
        document_text = (
            read_input_file(args.file)
            if args.file
            else read_interactive_input(args.type)
        )

        if args.type == "invoice":
            result = extract_invoice(document_text)
            result_json = invoice_to_json(result)
        else:
            result = extract_resume(document_text)
            result_json = resume_to_json(result)

        print(f"\n=== Extracted {args.type.title()} JSON ===")
        print(result_json)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result_json + "\n", encoding="utf-8")
            print(f"\nJSON saved to: {output_path.resolve()}")

    except KeyboardInterrupt:
        print("\nExtraction cancelled by user.")
        sys.exit(130)

    except (FileNotFoundError, ValueError) as exc:
        print(f"\nError: {exc}")
        sys.exit(1)

    except Exception as exc:
        print(f"\nExtraction failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
