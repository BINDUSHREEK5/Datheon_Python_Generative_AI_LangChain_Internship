from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app import build_prompt, build_retrieval_query, process_uploaded_documents, stream_answer


def test_build_prompt_uses_context_only():
    docs = [
        SimpleNamespace(
            page_content="Datheon focuses on data, AI, and automation.",
            metadata={"source_filename": "example.txt", "file_type": "txt"},
        )
    ]

    prompt = build_prompt("What does Datheon focus on?", docs)

    assert "Context:" in prompt
    assert "Datheon focuses on data, AI, and automation." in prompt
    assert "What does Datheon focus on?" in prompt


def test_build_retrieval_query_uses_recent_user_history_for_followups():
    history = [
        {"role": "user", "content": "Tell me about Datheon."},
        {"role": "assistant", "content": "Datheon focuses on data, AI, and automation."},
        {"role": "user", "content": "Explain the second one."},
    ]

    query = build_retrieval_query("Explain the second one.", history)

    assert "Tell me about Datheon." in query
    assert "Explain the second one." in query
    assert "assistant" not in query.lower()


def test_stream_answer_uses_retrieved_content_and_returns_citations():
    docs = [
        SimpleNamespace(
            page_content="Datheon focuses on data, AI, and automation.",
            metadata={"source_filename": "example.pdf", "file_type": "pdf", "page_number": 2},
        )
    ]

    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search.return_value = docs

    with patch("app.ChatGoogleGenerativeAI") as mock_llm:
        mock_llm.return_value.stream.return_value = [
            SimpleNamespace(content="Datheon focuses on data, AI, and automation."),
        ]

        chunks, citations = stream_answer(
            "What does Datheon focus on?",
            mock_vectorstore,
            api_key="test-key",
            model_name="gemini-test",
        )

    assert chunks == ["Datheon focuses on data, AI, and automation."]
    assert citations == ["example.pdf (Page 2)"]


def test_stream_answer_returns_missing_information_message_when_no_docs():
    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search.return_value = []

    chunks, citations = stream_answer(
        "What is the capital of France?",
        mock_vectorstore,
        api_key="test-key",
        model_name="gemini-test",
    )

    assert chunks == ["The uploaded documents do not contain enough information to answer this question."]
    assert citations == []


def test_process_uploaded_documents_stores_original_names_and_resets_history():
    class FakeUpload:
        def __init__(self, name, payload):
            self.name = name
            self._payload = payload

        def getvalue(self):
            return self._payload

    uploaded = [FakeUpload("example.txt", b"Datheon focuses on data, AI, and automation.")]
    fake_doc = SimpleNamespace(
        page_content="Datheon focuses on data, AI, and automation.",
        metadata={"source_filename": "example.txt", "file_type": "txt"},
    )

    with patch("app.load_document", return_value=[fake_doc]), patch("app.RAGPipeline") as mock_pipeline:
        mock_pipeline.return_value.create_vector_store.return_value = MagicMock()
        st_session = SimpleNamespace(messages=[], vectorstore=None, uploaded_files=[], rag_pipeline=None)
        with patch("app.st.session_state", st_session, create=True):
            process_uploaded_documents(uploaded)

    assert st_session.uploaded_files == ["example.txt"]
    assert st_session.messages == []
    assert st_session.vectorstore is not None


def test_build_prompt_ignores_prior_assistant_message_as_evidence():
    docs = [
        SimpleNamespace(
            page_content="Datheon focuses on data, AI, and automation.",
            metadata={"source_filename": "example.txt", "file_type": "txt"},
        )
    ]
    history = [
        {"role": "assistant", "content": "Use the document context."},
        {"role": "user", "content": "Explain the first one."},
    ]

    prompt = build_prompt("Explain the first one.", docs, conversation_history=history)

    assert "Assistant: Use the document context." in prompt
    assert "Explain the first one." in prompt
    assert "Context:" in prompt
