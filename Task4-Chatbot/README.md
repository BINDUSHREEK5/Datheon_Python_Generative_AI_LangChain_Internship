# Datheon AI Chatbot

This project is a beginner-friendly chatbot built with Streamlit and the existing Task 2 RAG pipeline. It reuses the working Task 2 document loader, chunking, embedding, retrieval, and citation logic, then adds a conversational interface with message history and streaming responses.

## Project overview

The chatbot lets a user upload PDF and TXT files, index the documents with the existing RAG pipeline, and ask questions in a chat-style interface. The app keeps the conversation history in session state, streams the model response in real time, and shows citations for the retrieved sources.

## Architecture

1. The user uploads PDF or TXT files in the sidebar.
2. The app reuses the Task 2 document loader and the Task 2 `RAGPipeline`.
3. The uploaded documents are chunked, embedded, and stored in ChromaDB.
4. Each chat question triggers a similarity search in the current vector store.
5. The model is prompted with only the retrieved document snippets.
6. The response is streamed back to the Streamlit chat interface.
7. Citations are displayed for the source filename and PDF page numbers when applicable.

## Technologies used

- Python
- Streamlit
- LangChain
- LangChain Google GenAI
- ChromaDB
- FastEmbed embeddings
- Pytest

## Installation for Windows

1. Open PowerShell in `Task4-Chatbot`.
2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install the project dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Environment variables

Create a `.env` file in the Task 4 project root using the example below:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

If `GEMINI_API_KEY` is missing, the app will show an error before answering.

## How to run the chatbot

From the project folder:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Then open the local Streamlit URL in a browser.

## Example chat flow

1. Upload a PDF or TXT file.
2. Click `Process documents`.
3. Ask a question such as:

```text
What are the main themes discussed in the uploaded document?
```

4. The chatbot will answer using the retrieved document context and show source citations.

## How to run tests

```powershell
.\.venv\Scripts\Activate.ps1
pytest -q
```

## Project limitations

- The chatbot relies on the uploaded document set in the current session.
- The model answer is grounded in retrieved document context only.
- Real Gemini API usage requires a valid `GEMINI_API_KEY` and internet access.
- Streaming output depends on the underlying LangChain Google GenAI client behavior.
