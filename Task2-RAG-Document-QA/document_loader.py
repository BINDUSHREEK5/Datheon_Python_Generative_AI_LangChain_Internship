from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def _plain_text_documents_from_pdf(path: Path) -> list[Document]:
    """Read all pages from a PDF and keep page metadata human-readable."""
    reader = PdfReader(str(path))
    documents: list[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        metadata = {
            "source": str(path),
            "source_filename": path.name,
            "filename": path.name,
            "file_type": "pdf",
            "page": page_number - 1,
            "page_number": page_number,
        }
        documents.append(Document(page_content=text.strip(), metadata=metadata))

    return documents


def _plain_text_documents_from_txt(path: Path) -> list[Document]:
    """Read a TXT file and keep filename metadata without inventing pages."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []

    return [
        Document(
            page_content=text.strip(),
            metadata={
                "source": str(path),
                "source_filename": path.name,
                "filename": path.name,
                "file_type": "txt",
            },
        )
    ]


def load_document(file_path: str) -> list[Document]:
    """Load a PDF or TXT document and return LangChain Document objects."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _plain_text_documents_from_pdf(path)

    if suffix == ".txt":
        return _plain_text_documents_from_txt(path)

    raise ValueError("Unsupported file type. Please upload a PDF or TXT file.")


def load_documents(file_paths: list[str]) -> list[Document]:
    """Load multiple supported documents and flatten the result."""
    if not file_paths:
        return []

    all_documents: list[Document] = []
    for file_path in file_paths:
        all_documents.extend(load_document(file_path))
    return all_documents
