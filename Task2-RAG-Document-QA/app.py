import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from document_loader import load_document
from rag_pipeline import RAGPipeline


load_dotenv()


def initialize_session_state() -> None:
    """Initialize the application session state."""
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline(collection_name=f"rag-session-{uuid.uuid4().hex}")
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = ""
    if "last_citations" not in st.session_state:
        st.session_state.last_citations = []


def reset_session_documents() -> None:
    """Clear any stale uploaded content before indexing a new upload session."""
    st.session_state.vectorstore = None
    st.session_state.rag_pipeline = RAGPipeline(collection_name=f"rag-session-{uuid.uuid4().hex}")
    st.session_state.uploaded_files = []
    st.session_state.last_answer = ""
    st.session_state.last_citations = []


def persist_uploaded_files(uploaded_files):
    """Save uploaded files to a temp directory and return their paths."""
    saved_paths: list[str] = []

    for uploaded_file in uploaded_files:
        file_name = Path(uploaded_file.name)
        file_suffix = file_name.suffix.lower()

        if file_suffix not in {".pdf", ".txt"}:
            raise ValueError(f"Unsupported file type: {file_name.suffix}. Please upload PDF or TXT files only.")

        target_dir = Path(tempfile.gettempdir()) / "datheon-rag-uploads"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{uuid.uuid4().hex}_{file_name.name}"
        target_path.write_bytes(uploaded_file.getvalue())
        saved_paths.append(str(target_path))

    return saved_paths


def process_uploaded_documents(uploaded_files):
    """Load, split, embed, and index the uploaded document set."""
    if not uploaded_files:
        raise ValueError("Please upload at least one PDF or TXT file.")

    reset_session_documents()

    saved_paths = persist_uploaded_files(uploaded_files)
    documents = []
    for saved_path in saved_paths:
        documents.extend(load_document(saved_path))

    if not documents:
        raise ValueError("No readable content was found in the uploaded files.")

    pipeline = RAGPipeline(collection_name=f"rag-session-{uuid.uuid4().hex}")
    vectorstore = pipeline.create_vector_store(documents)
    st.session_state.rag_pipeline = pipeline
    st.session_state.vectorstore = vectorstore
    st.session_state.uploaded_files = [Path(path).name for path in saved_paths]
    return pipeline, vectorstore


def main() -> None:
    st.set_page_config(page_title="Datheon RAG QA", page_icon="📚")
    initialize_session_state()

    st.title("Datheon RAG Document Q&A")
    st.caption("Ask questions about your uploaded PDFs and TXT files.")

    with st.sidebar:
        st.subheader("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )

        process_button = st.button("Process documents", type="primary")
        if process_button:
            try:
                with st.spinner("Loading and indexing your documents..."):
                    process_uploaded_documents(uploaded_files)
                st.success("Documents processed successfully.")
            except Exception as exc:  # pragma: no cover - UI error path
                st.error(f"Processing failed: {exc}")

    st.write("Upload your documents, then ask a question about the content.")

    if st.session_state.vectorstore is None:
        st.info("No processed documents yet. Upload PDFs or TXT files and click Process documents.")
        return

    question = st.text_input("Ask a question about the uploaded files", key="question_input")
    submit_button = st.button("Submit question")

    if submit_button and question.strip():
        try:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            if not gemini_api_key:
                raise ValueError("GEMINI_API_KEY is missing. Add your key to the environment or .env file.")

            with st.spinner("Searching the document store and generating an answer..."):
                result = st.session_state.rag_pipeline.answer_question(
                    question,
                    vectorstore=st.session_state.vectorstore,
                    api_key=gemini_api_key,
                    model_name=gemini_model,
                )

            st.session_state.last_answer = result["answer"]
            st.session_state.last_citations = result["citations"]
            st.subheader("Answer")
            st.write(result["answer"])

            if result["citations"]:
                st.subheader("Sources")
                for citation in result["citations"]:
                    st.write(f"- {citation}")
        except Exception as exc:  # pragma: no cover - UI error path
            st.error(f"An error occurred while generating the answer: {exc}")

    if st.session_state.last_answer:
        st.subheader("Latest answer")
        st.write(st.session_state.last_answer)
        if st.session_state.last_citations:
            st.subheader("Latest citations")
            for citation in st.session_state.last_citations:
                st.write(f"- {citation}")


if __name__ == "__main__":
    main()
