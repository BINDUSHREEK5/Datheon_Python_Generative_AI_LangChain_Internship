import os
import sys
import tempfile
import traceback
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK2_DIR = PROJECT_ROOT / "Task2-RAG-Document-QA"
if str(TASK2_DIR) not in sys.path:
    sys.path.insert(0, str(TASK2_DIR))

from document_loader import load_document
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_pipeline import RAGPipeline, format_citations


load_dotenv(TASK2_DIR / ".env")


def apply_chat_ui_theme() -> None:
    """Apply the selected light/dark palette to the chatbot workspace."""
    theme = st.session_state.get("theme_mode", "Light")

    if theme == "Dark":
        palette = {
            "page_bg": "#111111",
            "app_bg": "#151515",
            "sidebar_bg": "#1B1B1B",
            "surface_bg": "#353333C7",
            "card_bg": "#1D1D1D",
            "chat_user_bg": "#282828",
            "chat_assistant_bg": "#1D1D1D",
            "input_bg": "#1B1B1B",
            "text": "#EEEEEE",
            "muted_text": "#B5B5B5",
            "heading": "#FAFAFA",
            "border": "#383838",
            "accent": "#FFFFFF",
            "accent_hover": "#DEDEDE",
            "button_text": "#171717",
            "shadow": "rgba(0, 0, 0, 0.18)",
            "warning_bg": "#262626",
            "error_bg": "#292929",
            "success_bg": "#242424",
            "placeholder": "#A0A0A0",
            "file_uploader_bg": "#202020",
        }
    else:
        palette = {
            "page_bg": "#F3F3F3",
            "app_bg": "#F3F3F3",
            "sidebar_bg": "#EAEAEA",
            "surface_bg": "#FFFFFF",
            "card_bg": "#FFFFFF",
            "chat_user_bg": "#F0F0F0",
            "chat_assistant_bg": "#FFFFFF",
            "input_bg": "#FFFFFF",
            "text": "#242424",
            "muted_text": "#666666",
            "heading": "#171717",
            "border": "#D8D8D8",
            "accent": "#252525",
            "accent_hover": "#404040",
            "button_text": "#FFFFFF",
            "shadow": "rgba(0, 0, 0, 0.06)",
            "warning_bg": "#F1F1F1",
            "error_bg": "#EEEEEE",
            "success_bg": "#EFEFEF",
            "placeholder": "#777777",
            "file_uploader_bg": "#F7F7F7",
        }

    st.markdown(
        f"""
        <style>
        :root {{
            color-scheme: {"dark" if theme == "Dark" else "light"};
        }}
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp {{
            background: {palette['page_bg']} !important;
            color: {palette['text']};
            font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        }}
        .block-container {{
            max-width: 1080px;
            padding: 2.5rem 2rem 3rem;
        }}
        [data-testid="stSidebar"], [data-testid="stSidebar"] > div {{
            background: {palette['sidebar_bg']};
            color: {palette['text']};
        }}
        [data-testid="stSidebar"] > div {{
            border-right: 1px solid {palette['border']};
            padding: 1.25rem 1rem;
        }}
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6 {{
            color: {palette['heading']} !important;
            margin-top: 0.8rem;
            margin-bottom: 0.65rem;
            letter-spacing: -0.02em;
        }}
        [data-testid="stSidebar"] h3 {{
            padding-bottom: 0.55rem;
            border-bottom: 1px solid {palette['border']};
        }}
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] li,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stRadio {{
            color: {palette['text']} !important;
        }}
        [data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child {{
            background: transparent !important;
            border: 1px solid {palette['border']} !important;
        }}
        [data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) > div:first-child {{
            background: {palette['accent']} !important;
            border-color: {palette['accent']} !important;
        }}
        [data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) > div:first-child > div {{
            background: {palette['button_text']} !important;
        }}
        [data-testid="stSidebar"] .stRadio [role="radio"] span,
        [data-testid="stSidebar"] .stRadio label {{
            color: {palette['text']} !important;
        }}
        [data-testid="stSidebar"] [data-testid="stFileUploader"] {{
            background: {palette['surface_bg']};
            border: 1px solid {palette['border']};
            border-radius: 14px;
            padding: 0.55rem;
            box-shadow: 0 2px 8px {palette['shadow']};
        }}
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
            background: {palette['file_uploader_bg']};
            border: 1px dashed {palette['border']};
            border-radius: 10px;
            color: {palette['text']} !important;
            min-height: 120px;
            padding: 0.9rem 0.75rem;
        }}
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {{
            color: {palette['text']} !important;
        }}
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] .stFileUploaderLabel,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] .stFileUploaderHelp,
        [data-testid="stSidebar"] [data-testid="stFileUploaderFileName"],
        [data-testid="stSidebar"] [data-testid="stFileUploaderFileSize"],
        [data-testid="stSidebar"] .stFileUploader > div > div > div {{
            color: {palette['muted_text']} !important;
        }}
        [data-testid="stSidebar"] .stFileUploader button,
        [data-testid="stSidebar"] .stFileUploader .uploadButton {{
            background: {palette['accent']} !important;
            color: {palette['button_text']} !important;
            border-color: {palette['accent']} !important;
        }}
        [data-testid="stSidebar"] .stFileUploader button:hover,
        [data-testid="stSidebar"] .stFileUploader .uploadButton:hover {{
            background: {palette['accent_hover']} !important;
            border-color: {palette['accent_hover']} !important;
        }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
            color: {palette['heading']};
            letter-spacing: -0.02em;
        }}
        .stApp p, .stApp li {{
            color: {palette['text']};
        }}
        .stApp h1 {{
            font-size: clamp(2rem, 4vw, 2.7rem);
            line-height: 1.12;
            letter-spacing: -0.045em;
            margin-bottom: 0.3rem;
        }}
        .workspace-kicker {{
            color: {palette['muted_text']};
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.7rem;
        }}
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stCaptionContainer"] p {{
            color: {palette['muted_text']} !important;
        }}
        [data-testid="stHeader"] {{
            background: {palette['page_bg']} !important;
            border-bottom: 1px solid {palette['border']};
        }}
        [data-testid="stToolbar"], [data-testid="stToolbar"] *,
        [data-testid="stHeader"] button, [data-testid="stHeader"] a,
        [data-testid="stSidebarCollapsedControl"] button {{
            color: {palette['text']} !important;
        }}
        [data-testid="stHeader"] button:hover,
        [data-testid="stToolbar"] button:hover,
        [data-testid="stSidebarCollapsedControl"] button:hover {{
            background: {palette['surface_bg']} !important;
            color: {palette['heading']} !important;
        }}
        [data-testid="stToolbar"] svg, [data-testid="stHeader"] svg,
        [data-testid="stSidebarCollapsedControl"] svg {{
            fill: {palette['text']} !important;
            color: {palette['text']} !important;
        }}
        [data-testid="stDeployButton"] {{
            color: {palette['text']} !important;
            border: 1px solid {palette['border']} !important;
            border-radius: 8px !important;
            background: {palette['surface_bg']} !important;
        }}
        .stButton > button {{
            border-radius: 10px;
            background: {palette['surface_bg']};
            color: {palette['text']};
            border: 1px solid {palette['border']};
            font-weight: 600;
            min-height: 2.5rem;
            box-shadow: 0 1px 3px {palette['shadow']};
            transition: background-color 0.18s ease, border-color 0.18s ease, opacity 0.18s ease;
        }}
        .stButton > button:hover {{
            background: {palette['sidebar_bg']};
            border-color: {palette['border']};
            color: {palette['text']};
        }}
        .stButton > button:disabled {{
            opacity: 0.55;
            cursor: not-allowed;
            background: {palette['surface_bg']};
            border-color: {palette['border']};
            color: {palette['muted_text']};
        }}
        .stFileUploaderDropzone {{
            background: {palette['file_uploader_bg']};
            border: 1px dashed {palette['border']};
            border-radius: 10px;
            color: {palette['text']} !important;
            min-height: 110px;
        }}
        [data-testid="stFileUploaderFileName"], [data-testid="stFileUploaderFileSize"],
        .stFileUploader > div > div > div, .stFileUploaderHelp {{
            color: {palette['muted_text']} !important;
        }}
        .stChatInput {{
            border: 1px solid {palette['border']};
            border-radius: 12px;
            background: {palette['input_bg']};
            box-shadow: 0 2px 8px {palette['shadow']};
        }}
        [data-testid="stChatInput"] textarea {{
            color: {palette['text']} !important;
            caret-color: {palette['text']};
        }}
        [data-testid="stChatInput"] textarea::placeholder {{
            color: {palette['placeholder']} !important;
        }}
        .stChatMessage {{
            margin-bottom: 0.75rem;
        }}
        [data-testid="stChatMessageContent"] {{
            background: {palette['chat_assistant_bg']};
            border: 1px solid {palette['border']};
            border-radius: 14px;
            padding: 0.95rem 1.1rem;
            box-shadow: 0 2px 8px {palette['shadow']};
        }}
        [data-testid="stChatMessage"]:nth-of-type(odd) [data-testid="stChatMessageContent"] {{
            background: {palette['chat_user_bg']};
        }}
        .stAlert, div[data-testid="stAlert"] {{
            background: {palette['surface_bg']};
            border: 1px solid {palette['border']};
            color: {palette['text']};
            border-radius: 12px;
        }}
        .stSuccess {{
            background: {palette['success_bg']};
            border-color: {palette['border']};
            color: {palette['text']};
        }}
        .stWarning {{
            background: {palette['warning_bg']};
            border-color: {palette['border']};
            color: {palette['text']};
        }}
        .stError {{
            background: {palette['error_bg']};
            border-color: {palette['border']};
            color: {palette['text']};
        }}
        .welcome-card {{
            background: {palette['card_bg']};
            border: 1px solid {palette['border']};
            border-radius: 18px;
            padding: clamp(1.5rem, 4vw, 2.5rem);
            box-shadow: 0 8px 28px {palette['shadow']};
            max-width: 760px;
            margin: 4vh auto 0;
        }}
        .welcome-eyebrow {{
            color: {palette['muted_text']};
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 1.2rem;
        }}
        .welcome-card h3 {{
            margin: 0 0 0.75rem 0;
            color: {palette['heading']};
            font-size: clamp(1.6rem, 3vw, 2.1rem);
            letter-spacing: -0.035em;
        }}
        .welcome-card p {{
            margin: 0;
            color: {palette['muted_text']};
            font-size: 1.02rem;
            line-height: 1.7;
            max-width: 600px;
        }}
        .welcome-rule {{
            width: 42px;
            height: 3px;
            margin: 1.4rem 0;
            background: {palette['accent']};
            border-radius: 3px;
        }}
        .welcome-note {{
            padding-top: 1rem;
            border-top: 1px solid {palette['border']};
            color: {palette['muted_text']};
            font-size: 0.88rem;
        }}
        .citation-box {{
            background: {palette['card_bg']};
            border: 1px solid {palette['border']};
            border-left: 3px solid {palette['accent']};
            border-radius: 10px;
            padding: 0.65rem 0.9rem;
            margin-top: 0.5rem;
            box-shadow: 0 1px 4px {palette['shadow']};
        }}
        .citation-box ul {{
            margin: 0;
            padding-left: 1.15rem;
        }}
        .citation-box li, .citation-box p, .citation-box strong {{
            color: {palette['text']};
        }}
        .stRadio > div, .stRadio label, .stRadio span {{
            color: {palette['text']} !important;
        }}
        .stTextInput > div > div > input, .stTextArea > div > div > textarea {{
            background: {palette['input_bg']};
            color: {palette['text']};
            border: 1px solid {palette['border']};
            border-radius: 10px;
        }}
        @media (max-width: 700px) {{
            .block-container {{
                padding: 1.5rem 1rem 2rem;
            }}
            .welcome-card {{
                margin-top: 2vh;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    """Initialize chat state and the existing Task 2 RAG pipeline."""
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "Light"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline(collection_name=f"chatbot-session-{uuid.uuid4().hex}")
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []


def save_uploaded_files(uploaded_files):
    """Store uploaded files on disk so the Task 2 document loader can read them."""
    saved_paths = []
    target_dir = Path(tempfile.gettempdir()) / "datheon-chatbot-uploads"
    target_dir.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:
        file_name = Path(uploaded_file.name)
        file_suffix = file_name.suffix.lower()
        if file_suffix not in {".pdf", ".txt"}:
            raise ValueError(f"Unsupported file type: {file_name.suffix}. Please upload PDF or TXT files only.")

        target_path = target_dir / f"{uuid.uuid4().hex}_{file_name.name}"
        target_path.write_bytes(uploaded_file.getvalue())
        saved_paths.append(str(target_path))

    return saved_paths


def process_uploaded_documents(uploaded_files) -> None:
    """Load, index, and store uploaded PDFs/TXT for this chatbot session."""
    if not uploaded_files:
        raise ValueError("Please upload at least one PDF or TXT file.")

    saved_paths = save_uploaded_files(uploaded_files)
    documents = []
    for saved_path in saved_paths:
        documents.extend(load_document(saved_path))

    if not documents:
        raise ValueError("No readable content was found in the uploaded files.")

    st.session_state.rag_pipeline = RAGPipeline(collection_name=f"chatbot-session-{uuid.uuid4().hex}")
    st.session_state.vectorstore = st.session_state.rag_pipeline.create_vector_store(documents)
    st.session_state.uploaded_files = [uploaded_file.name for uploaded_file in uploaded_files]
    st.session_state.messages = []


def build_context(retrieved_docs) -> str:
    """Create a grounded context block from retrieved document chunks."""
    if not retrieved_docs:
        return ""

    context_parts = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        metadata = doc.metadata or {}
        source_name = metadata.get("source_filename") or Path(metadata.get("source", "unknown")).name
        page_number = metadata.get("page_number")
        page_text = f" (Page {page_number})" if metadata.get("file_type") == "pdf" and page_number is not None else ""
        context_parts.append(f"[Document {idx}] Source: {source_name}{page_text}\n{doc.page_content.strip()}")
    return "\n\n".join(context_parts)


def _format_conversation_history(messages) -> str:
    """Return a compact conversation summary without treating prior assistant replies as evidence."""
    if not messages:
        return ""

    history_lines = ["Conversation history (for follow-up context only; not as evidence):"]
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if not content:
            continue
        if role == "assistant":
            history_lines.append(f"Assistant: {content}")
        else:
            history_lines.append(f"User: {content}")
    return "\n".join(history_lines)


def build_retrieval_query(question: str, conversation_history=None) -> str:
    """Expand follow-up questions using prior user messages without treating older assistant replies as evidence."""
    previous_user_messages = []
    for msg in conversation_history or []:
        if msg.get("role") == "user" and msg.get("content"):
            previous_user_messages.append(msg["content"].strip())

    if not previous_user_messages:
        return question.strip()

    recent_history = previous_user_messages[-2:]
    if recent_history:
        return " ".join(recent_history + [question.strip()])
    return question.strip()


def build_prompt(question: str, retrieved_docs, conversation_history=None) -> str:
    """Construct a grounded prompt using only the retrieved document context."""
    context = build_context(retrieved_docs)
    history_text = _format_conversation_history(conversation_history or [])
    if not context:
        prompt = (
            "You are a document-grounded assistant. Use only the retrieved document chunks as factual evidence. "
            "Previous conversation history is only for follow-up context and is not evidence. "
            "If the answer is not explicitly supported by the uploaded documents, reply exactly: "
            "The uploaded documents do not contain enough information to answer this question.\n\n"
            f"{history_text}\n\nQuestion:\n{question}\n\nAnswer:"
        )
        return prompt

    prompt = (
        "You are a document-grounded assistant. Use only the retrieved document chunks as factual evidence. "
        "Previous conversation history is only for follow-up context and is not evidence. "
        "If the answer is not explicitly supported by the uploaded documents, reply exactly: "
        "The uploaded documents do not contain enough information to answer this question.\n\n"
        f"{history_text}\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )
    return prompt


def stream_answer(question: str, vectorstore, api_key: str, model_name: str, conversation_history=None):
    """Return the full answer chunks and citations for compatibility with the existing tests."""
    if vectorstore is None:
        raise ValueError("Please process documents before asking a question.")

    retrieval_question = build_retrieval_query(question, conversation_history)
    retrieved_docs = vectorstore.similarity_search(retrieval_question, k=5)
    if not retrieved_docs:
        fallback = "The uploaded documents do not contain enough information to answer this question."
        return [fallback], []

    llm = ChatGoogleGenerativeAI(model=model_name, api_key=api_key, temperature=0)
    prompt = build_prompt(question, retrieved_docs, conversation_history=conversation_history)

    chunks = []
    for chunk in llm.stream(prompt):
        text = getattr(chunk, "content", "")
        if isinstance(text, list):
            text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
        elif text is None:
            text = ""
        text = str(text).strip()
        if text:
            chunks.append(text)

    if not chunks:
        answer = "The uploaded documents do not contain enough information to answer this question."
        return [answer], []

    answer = "".join(chunks).strip()
    if answer.lower().startswith("the uploaded documents do not contain enough information"):
        answer = "The uploaded documents do not contain enough information to answer this question."
        return [answer], []

    return chunks, format_citations(retrieved_docs)


def stream_answer_chunks(question: str, vectorstore, api_key: str, model_name: str, conversation_history=None):
    """Yield Gemini chunks directly to Streamlit while preserving the final answer text and citations."""
    if vectorstore is None:
        raise ValueError("Please process documents before asking a question.")

    retrieval_question = build_retrieval_query(question, conversation_history)
    retrieved_docs = vectorstore.similarity_search(retrieval_question, k=5)
    if not retrieved_docs:
        fallback = "The uploaded documents do not contain enough information to answer this question."
        return iter([fallback]), []

    llm = ChatGoogleGenerativeAI(model=model_name, api_key=api_key, temperature=0)
    prompt = build_prompt(question, retrieved_docs, conversation_history=conversation_history)

    def stream_iterator():
        for chunk in llm.stream(prompt):
            text = getattr(chunk, "content", "")
            if isinstance(text, list):
                text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
            elif text is None:
                text = ""
            text = str(text).strip()
            if text:
                yield text

    return stream_iterator(), format_citations(retrieved_docs)


def render_citations(citations):
    """Display a tidy citation list for the answer."""
    if not citations:
        return

    st.markdown("### Sources")
    citation_html = "<div class='citation-box'><ul>"
    for citation in citations:
        citation_html += f"<li>{citation}</li>"
    citation_html += "</ul></div>"
    st.markdown(citation_html, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Datheon Chatbot", page_icon="🤖")
    initialize_session_state()

    st.markdown("<div class='workspace-kicker'>Datheon · Document research</div>", unsafe_allow_html=True)
    st.title("Datheon AI Chatbot")
    st.caption("A focused workspace for exploring your documents and getting source-grounded answers.")

    with st.sidebar:
        st.subheader("Theme")
        selected_theme = st.radio(
            "Appearance",
            options=["Light", "Dark"],
            index=0 if st.session_state.theme_mode == "Light" else 1,
            key="theme_selector",
            horizontal=False,
        )
        st.session_state.theme_mode = selected_theme
        apply_chat_ui_theme()

        st.subheader("Documents")
        uploaded_files = st.file_uploader(
            "Upload PDFs or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload the documents you want the chatbot to use as context.",
        )

        if st.button("Process documents", use_container_width=True):
            try:
                with st.spinner("Loading and indexing your documents..."):
                    process_uploaded_documents(uploaded_files)
                st.success("Documents processed successfully.")
            except Exception as exc:
                st.error(f"Processing failed: {exc}")

        if st.session_state.uploaded_files:
            st.markdown("### Current files")
            for file_name in st.session_state.uploaded_files:
                st.write(f"- {file_name}")

    if st.session_state.vectorstore is None:
        st.markdown(
            """
            <div class='welcome-card'>
                <div class='welcome-eyebrow'>Document research workspace</div>
                <h3>Welcome to the Datheon AI Chatbot</h3>
                <p>Bring your documents into the conversation. Upload and process a PDF or TXT file, then ask questions and review answers with their source citations.</p>
                <div class='welcome-rule'></div>
                <div class='welcome-note'>Your answers are grounded in the documents you process for this session.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                render_citations(message["citations"])

    prompt = st.chat_input("Ask a question about the uploaded documents")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt, "citations": []})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            api_key = os.getenv("GEMINI_API_KEY")
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is missing. Add it to your environment or .env file.")

            stream_iterator, citations = stream_answer_chunks(
                question=prompt,
                vectorstore=st.session_state.vectorstore,
                api_key=api_key,
                model_name=model_name,
                conversation_history=st.session_state.messages[:-1],
            )

            accumulated_answer = ""
            placeholder = st.empty()

            with st.chat_message("assistant"):
                for chunk in stream_iterator:
                    accumulated_answer += chunk
                    placeholder.markdown(accumulated_answer)

                render_citations(citations)

            st.session_state.messages.append({
                "role": "assistant",
                "content": accumulated_answer.strip(),
                "citations": citations,
            })
        except Exception as exc:
            traceback.print_exc()
            with st.chat_message("assistant"):
                st.error(f"An error occurred while answering: {exc}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"An error occurred while answering: {exc}",
                "citations": [],
            })


if __name__ == "__main__":
    main()
