import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from document_loader import load_document
from rag_pipeline import RAGPipeline


load_dotenv()


def apply_ui_theme() -> None:
    """Apply a professional light/dark theme without disturbing the RAG session state."""
    theme = st.session_state.get("theme_mode", "Light")

    if theme == "Dark":
        page_bg = "#171b1d"
        surface_bg = "#20272b"
        content_bg = "#1a1f22"
        primary_text = "#f3f5f0"
        secondary_text = "#d9ded4"
        muted_text = "#b7c0b4"
        border = "#384245"
        accent = "#a2a86b"
        accent_soft = "#2f362a"
        input_bg = "#1d2326"
        info_bg = "#1d2326"
        warning_bg = "#29261e"
        sidebar_bg = "#20272b"
        sidebar_heading = "#f3f5f0"
        sidebar_text = "#d9ded4"
        sidebar_muted = "#b7c0b4"
    else:
        page_bg = "#f6f2ea"
        surface_bg = "#ffffff"
        content_bg = "#f9f7f3"
        primary_text = "#1f2320"
        secondary_text = "#2d332d"
        muted_text = "#58615b"
        border = "#d9d3cb"
        accent = "#5f6d3a"
        accent_soft = "#edf0d9"
        input_bg = "#ffffff"
        info_bg = "#f2f6ef"
        warning_bg = "#f8f1e2"
        sidebar_bg = "#1f2326"
        sidebar_heading = "#f5f6f2"
        sidebar_text = "#dfe4d9"
        sidebar_muted = "#b9c0b4"

    st.markdown(
        f"""
        <style>
        html, body, [data-testid="stAppViewContainer"], .stApp {{
            background-color: {page_bg};
            color: {primary_text};
        }}
        .stApp {{
            max-width: 100%;
            overflow-x: hidden;
        }}
        .block-container {{
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 3rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }}
        div[data-testid="stSidebar"] > div {{
            background-color: {sidebar_bg};
            border-right: 1px solid {border};
            color: {sidebar_text};
        }}
        .stSidebar, .stSidebar * {{
            color: {sidebar_text};
        }}
        .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6,
        .stSidebar .stMarkdown h1, .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3,
        .stSidebar .stMarkdown h4, .stSidebar .stMarkdown h5, .stSidebar .stMarkdown h6 {{
            color: {sidebar_heading};
        }}
        .stSidebar .stRadio, .stSidebar .stMarkdown, .stSidebar .stFileUploader, .stSidebar .stButton {{
            color: {sidebar_text};
        }}
        .stSidebar div[data-testid="stFileUploaderDropzone"] {{
            background: {content_bg};
            border: 1px solid {border};
            color: {sidebar_text};
        }}
        .stSidebar .stMarkdown p, .stSidebar .stMarkdown li, .stSidebar label,
        .stSidebar [data-testid="stFileUploaderFileName"],
        .stSidebar [data-testid="stFileUploaderFileSize"] {{
            color: {sidebar_text};
        }}
        .stSidebar .stMarkdown p, .stSidebar .stMarkdown li, .stSidebar label, .stSidebar [data-testid="stFileUploaderFileName"], .stSidebar [data-testid="stFileUploaderFileSize"] {{
            opacity: 0.95;
        }}
        .stSidebar .stRadio label, .stSidebar .stRadio div, .stSidebar .stRadio span {{
            color: {sidebar_text};
        }}
        .stSidebar .stFileUploaderHelp, .stSidebar .stFileUploader > div > div > div,
        .stSidebar .stFileUploader .help, .stSidebar .stFileUploader .stHelp {{
            color: {sidebar_muted};
        }}
        .stButton > button {{
            background-color: {accent};
            color: #ffffff;
            border: 1px solid {accent};
            font-weight: 600;
            border-radius: 8px;
            min-height: 2.5rem;
        }}
        .stButton > button:hover {{
            background-color: {accent};
            color: #ffffff;
            border-color: {accent};
            opacity: 0.96;
        }}
        .stDownloadButton > button, .stSecondaryButton > button {{
            background-color: {surface_bg};
            border: 1px solid {border};
            color: {primary_text};
        }}
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div,
        .stNumberInput > div > div,
        .stMultiSelect > div > div {{
            background-color: {input_bg};
            color: {primary_text};
            border: 1px solid {border};
            border-radius: 8px;
        }}
        .stTextInput label, .stTextArea label, .stSelectbox label, .stNumberInput label {{
            color: {secondary_text};
            font-weight: 600;
        }}
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {{
            color: {primary_text};
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
            line-height: 1.3;
        }}
        .stMarkdown p, .stMarkdown li, .stMarkdown div {{
            color: {secondary_text};
        }}
        .stDataFrame, .stCodeBlock, .stExpander {{
            border: 1px solid {border};
            border-radius: 10px;
            background: {content_bg};
        }}
        .stAlert, div[data-testid="stAlert"] {{
            background: {info_bg};
            border: 1px solid {border};
            color: {primary_text};
        }}
        .stSuccess {{
            background: {info_bg};
            border: 1px solid {border};
        }}
        .stWarning {{
            background: {warning_bg};
            border: 1px solid {border};
            color: {primary_text};
        }}
        .stError {{
            background: #3a1d1d;
            border: 1px solid #8b5e5e;
            color: #f5d7d7;
        }}
        .stInfo {{
            background: {info_bg};
            border: 1px solid {border};
            color: {primary_text};
        }}
        .success-box, .info-box, .warning-box {{
            padding: 0.9rem 1rem;
            border-left: 4px solid {accent};
            border-radius: 8px;
            background: {info_bg};
            color: {primary_text};
            margin: 0.85rem 0;
            border: 1px solid {border};
            max-width: 100%;
        }}
        .success-box {{
            border-left-color: {accent};
        }}
        .info-box {{
            border-left-color: {accent};
        }}
        .warning-box {{
            border-left-color: #c08b2f;
        }}
        .citation-box {{
            background: {content_bg};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            margin-top: 0.8rem;
        }}
        .citation-box p, .citation-box li, .citation-box strong {{
            color: {primary_text};
        }}
        [data-testid="stFileUploaderFileName"] {{
            color: {primary_text};
        }}
        [data-testid="stFileUploaderFileSize"] {{
            color: {muted_text};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    """Initialize the application session state."""
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "Light"
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
    original_names = [uploaded_file.name for uploaded_file in uploaded_files]
    documents = []
    for saved_path in saved_paths:
        documents.extend(load_document(saved_path))

    if not documents:
        raise ValueError("No readable content was found in the uploaded files.")

    pipeline = st.session_state.rag_pipeline
    vectorstore = pipeline.create_vector_store(documents)
    st.session_state.vectorstore = vectorstore
    st.session_state.uploaded_files = original_names
    return pipeline, vectorstore


def render_citations(citations):
    """Display citation entries in a tidy format."""
    if not citations:
        return

    st.markdown("### Sources")
    citation_html = "<div class='citation-box'><ul>"
    for citation in citations:
        citation_html += f"<li>{citation}</li>"
    citation_html += "</ul></div>"
    st.markdown(citation_html, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Datheon RAG QA", page_icon="📚")
    initialize_session_state()
    apply_ui_theme()

    st.title("Datheon RAG Document Q&A")
    st.caption("Grounded answers from your uploaded PDFs and TXT files.")

    with st.sidebar:
        st.subheader("Theme")
        selected_theme = st.radio("Appearance", options=["Light", "Dark"], index=0 if st.session_state.theme_mode == "Light" else 1, horizontal=False, key="theme_selector")
        st.session_state.theme_mode = selected_theme
        apply_ui_theme()

        st.subheader("Document Upload")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload one or more PDF or TXT files to build a document-specific knowledge base.",
        )

        process_button = st.button("Process documents", type="primary")
        if process_button:
            try:
                with st.spinner("Loading and indexing your documents..."):
                    process_uploaded_documents(uploaded_files)
                st.success("Documents processed successfully.")
            except Exception as exc:  # pragma: no cover - UI error path
                st.error(f"Processing failed: {exc}")

        if st.session_state.uploaded_files:
            st.markdown("### Current files")
            for file_name in st.session_state.uploaded_files:
                st.write(f"- {file_name}")

        clear_button = st.button("Clear current session")
        if clear_button:
            reset_session_documents()
            st.rerun()

    if st.session_state.vectorstore is None:
        st.markdown(
            """
            <div class="info-box">
                No processed documents yet. Upload PDFs or TXT files and click <strong>Process documents</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        """
        <div class="success-box">
            Document set ready. Ask a question about the current upload session only.
        </div>
        """,
        unsafe_allow_html=True,
    )

    question = st.text_input("Ask a question about the uploaded files", key="question_input")
    submit_button = st.button("Submit question", use_container_width=True)

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
        except Exception as exc:  # pragma: no cover - UI error path
            st.error(f"An error occurred while generating the answer: {exc}")

    if st.session_state.last_answer:
        st.markdown("### Answer")
        st.write(st.session_state.last_answer)
        render_citations(st.session_state.last_citations)


if __name__ == "__main__":
    main()
