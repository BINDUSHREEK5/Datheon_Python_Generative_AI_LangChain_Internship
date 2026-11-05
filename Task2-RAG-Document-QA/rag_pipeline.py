import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI


class RAGPipeline:
    """Create a simple retrieval-augmented generation pipeline."""

    def __init__(
        self,
        collection_name: str | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        persist_directory: str | None = None,
    ) -> None:
        load_dotenv()
        self.collection_name = collection_name or f"rag-session-{uuid.uuid4().hex}"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory or os.path.join(
            tempfile.gettempdir(), "datheon-rag", self.collection_name
        )
        self.vectorstore = None

    def clear_persisted_data(self) -> None:
        """Remove any stale Chroma data for this session before indexing new uploads."""
        self.vectorstore = None
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory, ignore_errors=True)

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split loaded documents into manageable chunks while preserving metadata."""
        if not documents:
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        chunks = splitter.split_documents(documents)

        for chunk in chunks:
            metadata = dict(chunk.metadata or {})
            source = metadata.get("source") or metadata.get("source_filename") or "unknown"
            metadata["source_filename"] = Path(source).name
            metadata["filename"] = Path(source).name
            if metadata.get("file_type") == "pdf":
                page_value = metadata.get("page")
                if isinstance(page_value, int):
                    metadata["page_number"] = page_value + 1
                elif page_value is not None:
                    try:
                        metadata["page_number"] = int(page_value) + 1
                    except (TypeError, ValueError):
                        metadata["page_number"] = page_value
            chunk.metadata = metadata

        return chunks

    def create_vector_store(self, documents: list[Document]):
        """Create a Chroma vector store from chunks and embeddings."""
        if not documents:
            raise ValueError("No documents were provided for indexing.")

        self.clear_persisted_data()

        chunks = self.chunk_documents(documents)
        if not chunks:
            raise ValueError("No content could be extracted from the uploaded documents.")

        embeddings = FastEmbedEmbeddings(model_name=self.embedding_model)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )
        return self.vectorstore

    def retrieve(self, question: str, k: int = 5, vectorstore=None):
        """Return the most relevant retrieved chunks for a question."""
        selected_store = vectorstore or self.vectorstore
        if selected_store is None:
            raise ValueError("No vector store is available. Process documents first.")

        if not question or not question.strip():
            raise ValueError("A question is required.")

        return selected_store.similarity_search(question, k=k)

    def answer_question(
        self,
        question: str,
        vectorstore=None,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """Answer a question using retrieved document context only."""
        if not question or not question.strip():
            raise ValueError("A question is required.")

        selected_store = vectorstore or self.vectorstore
        if selected_store is None:
            raise ValueError("No vector store is available. Process documents first.")

        retrieved_docs = selected_store.similarity_search(question, k=5)
        if not retrieved_docs:
            return {
                "answer": "The uploaded documents do not contain enough information to answer this question.",
                "citations": [],
                "source_documents": [],
            }

        context_parts: list[str] = []
        for idx, doc in enumerate(retrieved_docs, start=1):
            metadata = doc.metadata or {}
            source_name = metadata.get("source_filename") or Path(metadata.get("source", "unknown")).name
            page_number = metadata.get("page_number")
            page_text = f" (Page {page_number})" if metadata.get("file_type") == "pdf" and page_number is not None else ""
            context_parts.append(
                f"[Document {idx}] Source: {source_name}{page_text}\n{doc.page_content.strip()}"
            )
        context = "\n\n".join(context_parts)

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing. Add it to your environment or .env file.")

        resolved_model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        llm = ChatGoogleGenerativeAI(model=resolved_model, api_key=api_key, temperature=0)

        prompt_text = (
            "You are a document-grounded assistant. Use only the provided context to answer the question. "
            "If the answer is not explicitly supported by the uploaded documents, reply exactly: "
            "The uploaded documents do not contain enough information to answer this question.\n\n"
            f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )
        answer = llm.invoke(prompt_text).content.strip()

        citations = format_citations(retrieved_docs)

        if answer.lower().startswith("the uploaded documents do not contain enough information"):
            answer = "The uploaded documents do not contain enough information to answer this question."

        return {
            "answer": answer,
            "citations": citations,
            "source_documents": retrieved_docs,
        }

    @staticmethod
    def format_citations(documents: list[Document]) -> list[str]:
        return format_citations(documents)


def format_citations(documents: list[Document]) -> list[str]:
    """Build a de-duplicated list of human-readable citations for the answer."""
    citation_map: dict[tuple[str, str], str] = {}

    for document in documents:
        metadata = document.metadata or {}
        source_name = metadata.get("source_filename") or Path(metadata.get("source", "unknown")).name
        file_type = metadata.get("file_type")
        page_number = metadata.get("page_number")

        if file_type == "pdf" and page_number is not None:
            visible = f"{source_name} (Page {page_number})"
            key = (source_name, str(page_number))
        else:
            visible = source_name
            key = (source_name, "txt")

        citation_map.setdefault(key, visible)

    return list(citation_map.values())
