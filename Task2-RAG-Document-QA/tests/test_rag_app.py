import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from document_loader import load_document
from rag_pipeline import RAGPipeline, format_citations


def test_load_txt_document(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Datheon builds AI-powered workflows.", encoding="utf-8")

    documents = load_document(str(file_path))

    assert len(documents) == 1
    assert documents[0].page_content.strip() == "Datheon builds AI-powered workflows."
    assert documents[0].metadata["source_filename"] == "sample.txt"
    assert "page_number" not in documents[0].metadata


def test_load_pdf_document_preserves_metadata(tmp_path):
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\nnot_a_real_pdf")

    with patch("document_loader.PdfReader") as mock_reader:
        mock_reader.return_value.pages = [SimpleNamespace(extract_text=lambda: "This document explains vector databases.")]
        documents = load_document(str(file_path))

    assert documents[0].metadata["source_filename"] == "sample.pdf"
    assert documents[0].metadata["page"] == 0
    assert documents[0].metadata["page_number"] == 1
    assert documents[0].page_content == "This document explains vector databases."


def test_load_document_unsupported_format(tmp_path):
    unsupported = tmp_path / "sample.md"
    unsupported.write_text("Unsupported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(str(unsupported))


def test_load_document_missing_file(tmp_path):
    missing = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        load_document(str(missing))


def test_chunk_documents_preserve_metadata():
    pipeline = RAGPipeline(chunk_size=20, chunk_overlap=5)
    doc = Document(
        page_content="Alpha beta gamma delta epsilon zeta eta theta iota",
        metadata={"source": "example.pdf", "page": 2, "file_type": "pdf"},
    )

    chunks = pipeline.chunk_documents([doc])

    assert chunks
    assert all(chunk.metadata["source_filename"] == "example.pdf" for chunk in chunks)
    assert all(chunk.metadata["page_number"] == 3 for chunk in chunks)


def test_create_vector_store_and_retrieve():
    pipeline = RAGPipeline(collection_name="test-collection", chunk_size=20, chunk_overlap=5)
    docs = [
        Document(page_content="Datheon uses Python and AI.", metadata={"source": "demo.txt", "source_filename": "demo.txt"}),
        Document(page_content="The weather is cloudy today.", metadata={"source": "weather.txt", "source_filename": "weather.txt"}),
    ]

    fake_store = MagicMock()
    fake_store.similarity_search.return_value = [
        Document(page_content="Datheon uses Python and AI.", metadata={"source": "demo.txt", "source_filename": "demo.txt", "file_type": "txt"})
    ]

    with patch.object(RAGPipeline, "chunk_documents", return_value=docs), patch("rag_pipeline.Chroma.from_documents") as mock_chroma:
        mock_chroma.return_value = fake_store
        vectorstore = pipeline.create_vector_store(docs)
        results = pipeline.retrieve("What technology does Datheon use?", k=2)

    assert vectorstore is fake_store
    assert results[0].page_content == "Datheon uses Python and AI."


def test_empty_documents_error():
    pipeline = RAGPipeline()

    with pytest.raises(ValueError, match="No documents were provided"):
        pipeline.create_vector_store([])


def test_answer_question_for_missing_information():
    pipeline = RAGPipeline()
    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search.return_value = [
        Document(page_content="The document says Datheon uses Python.", metadata={"source": "notes.txt", "source_filename": "notes.txt", "file_type": "txt"})
    ]

    with patch("rag_pipeline.ChatGoogleGenerativeAI") as mock_llm:
        mock_llm.return_value.invoke.return_value = SimpleNamespace(content="The uploaded documents do not contain enough information to answer this question.")
        result = pipeline.answer_question(
            "What is the capital of France?",
            vectorstore=mock_vectorstore,
            api_key="test-key",
            model_name="gemini-test",
        )

    assert result["answer"] == "The uploaded documents do not contain enough information to answer this question."
    assert result["citations"] == ["notes.txt"]


def test_citation_formatting_deduplicates_sources():
    docs = [
        Document(page_content="First", metadata={"source": "alpha.pdf", "source_filename": "alpha.pdf", "file_type": "pdf", "page": 0, "page_number": 1}),
        Document(page_content="Second", metadata={"source": "alpha.pdf", "source_filename": "alpha.pdf", "file_type": "pdf", "page": 0, "page_number": 1}),
        Document(page_content="Third", metadata={"source": "notes.txt", "source_filename": "notes.txt", "file_type": "txt"}),
    ]

    citations = format_citations(docs)

    assert citations == ["alpha.pdf (Page 1)", "notes.txt"]


def test_document_isolation_after_replacing_uploads(tmp_path):
    example_doc = Document(
        page_content="Datheon focuses on data, AI, and automation.",
        metadata={"source": "example.txt", "source_filename": "example.txt", "file_type": "txt"},
    )
    assignment_doc = Document(
        page_content="This assignment covers model deployment and testing.",
        metadata={"source": "assignment.pdf", "source_filename": "assignment.pdf", "file_type": "pdf", "page": 0, "page_number": 1},
    )

    pipeline = RAGPipeline(
        collection_name="isolation-session",
        persist_directory=str(tmp_path / "persisted-session"),
        chunk_size=50,
        chunk_overlap=10,
    )

    first_store = MagicMock()
    first_store.similarity_search.return_value = [example_doc, assignment_doc]

    second_store = MagicMock()
    second_store.similarity_search.return_value = [assignment_doc]

    with patch("rag_pipeline.FastEmbedEmbeddings"), patch("rag_pipeline.Chroma.from_documents") as mock_chroma:
        mock_chroma.side_effect = [first_store, second_store]

        pipeline.create_vector_store([example_doc, assignment_doc])
        combined_results = pipeline.retrieve("What are the three main areas Datheon focuses on?", vectorstore=first_store)
        assert any(doc.metadata.get("source_filename") == "example.txt" for doc in combined_results)

        pipeline.create_vector_store([assignment_doc])
        replacement_results = pipeline.retrieve("What are the three main areas Datheon focuses on?", vectorstore=second_store)
        assert all(doc.metadata.get("source_filename") != "example.txt" for doc in replacement_results)

        with patch("rag_pipeline.ChatGoogleGenerativeAI") as mock_llm:
            mock_llm.return_value.invoke.return_value = SimpleNamespace(
                content="The uploaded documents do not contain enough information to answer this question."
            )
            answer = pipeline.answer_question(
                "What are the three main areas Datheon focuses on?",
                vectorstore=second_store,
                api_key="test-key",
                model_name="gemini-test",
            )

        assert answer["answer"] == "The uploaded documents do not contain enough information to answer this question."
        assert answer["citations"] == ["assignment.pdf (Page 1)"]
