# Structured LLM Prompt Pipeline & JSON Extractor

A Python-based LLM application that extracts structured information from invoices and resumes using Google's Gemini API. It uses parameterized prompt templates to guide extraction, parses the model's JSON response, and validates the extracted data using Pydantic.

## Features

- Extract structured information from invoices and resumes
- Parameterized prompt templates for different document types
- Gemini API integration
- JSON response parsing and validation
- Strict Pydantic schema validation
- Handles missing information using null values
- Input validation and error handling
- Interactive text input and file-based text input
- Save extracted results as JSON files
- Automated tests using pytest

## Technologies

- Python
- Google Gemini API (`google-genai`)
- Pydantic
- pytest
- python-dotenv

## Project Structure

```text
Datheon_Python_Generative_AI_LangChain_Internship/
│
└── Task1-Structured-LLM-JSON-Extractor/
    │
    ├── main.py
    ├── extractor.py
    ├── prompt_templates.py
    ├── schemas.py
    ├── requirements.txt
    ├── .env.example
    ├── README.md
    │
    ├── sample_data/
    │   ├── invoice.txt
    │   └── resume.txt
    │
    └── tests/
        └── test_extractor.py
```

## Prerequisites

- Python 3.10 or later
- Git (optional, for cloning)
- A Google Gemini API key

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/BINDUSHREEK5/Datheon_Python_Generative_AI_LangChain_Internship.git
cd Datheon_Python_Generative_AI_LangChain_Internship/Task1-Structured-LLM-JSON-Extractor
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure the API key

Copy `.env.example` to `.env` and add your Gemini API key.

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash
```

Use a model supported by your Gemini API account and the application's current API implementation if the default is unavailable.

**Security:** Never commit your real `.env` file or API key to GitHub.

## Usage

### Option 1: Extract an invoice from a text file

```powershell
python main.py --type invoice --file sample_data/invoice.txt --output output/invoice.json
```

### Option 2: Extract a resume from a text file

```powershell
python main.py --type resume --file sample_data/resume.txt --output output/resume.json
```

### Option 3: Interactive input

Run:

```powershell
python main.py --type invoice
```

Paste the invoice text into the terminal. Type `END` on a separate line when finished.

For a resume, use:

```powershell
python main.py --type resume
```

The application extracts the information, validates the result, displays formatted JSON, and optionally saves it to the specified output file.

## Run the Tests

```powershell
python -m pytest tests -v
```

The test suite currently contains 15 test cases covering:

- Invoice and resume extraction
- JSON parsing and malformed responses
- Pydantic validation
- Missing required fields
- Invalid email addresses
- Unexpected fields
- Invalid invoice dates
- Empty input and missing API keys
- Gemini API failures
- Oversized input
- Markdown code-fenced JSON responses

The tests use mocked Gemini responses, so automated testing does not require live API calls.

## How It Works

1. The user provides invoice or resume text.
2. The application selects the corresponding parameterized prompt template.
3. The prompt and document text are sent to the Gemini API.
4. The model returns a JSON response.
5. The application parses the JSON response.
6. Pydantic validates the extracted information against the appropriate schema.
7. The validated result is displayed and optionally saved as a JSON file.

## Limitations

- Currently supports text-based input files (`.txt`), not direct PDF or image extraction.
- Extraction quality depends on the source text, Gemini model, and API availability.
- LLM-generated information should always be reviewed before being used for business or hiring decisions.

## Sample Input and Output

This section demonstrates how the application extracts structured information from invoice and resume text.

### 1. Invoice Extraction

**Sample Input:**

The application accepts invoice text like the following. The complete sample is available in `sample_data/invoice.txt`.

```text
Invoice Number: INV-104
Vendor: ACME Technologies
Invoice Date: 2026-10-12
Due Date: 2026-10-30
Total Amount: 2500 INR
```

**Sample Output:**

The extracted information is returned as structured JSON and validated using the Pydantic invoice schema.

```json
{
  "invoice_number": "INV-104",
  "vendor": "ACME Technologies",
  "invoice_date": "2026-10-12",
  "due_date": "2026-10-30",
  "total_amount": "2500",
  "currency": "INR"
}
```

### 2. Resume Extraction

**Sample Input:**

The application also accepts resume text. See `sample_data/resume.txt` for the complete sample resume.

**Sample Output:**

The resume information is extracted into structured JSON and validated using the Pydantic resume schema.

To generate the actual output, run:

```powershell
python main.py --type resume --file sample_data/resume.txt --output output/resume.json
```

The generated JSON contains the extracted resume fields defined in the Pydantic schema.

