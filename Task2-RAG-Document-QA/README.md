# Datheon RAG Document Q&A

This project is a beginner-friendly Retrieval-Augmented Generation (RAG) application built with Python, LangChain, ChromaDB, and Gemini. It lets a user upload PDF and TXT documents, index their contents, retrieve the most relevant chunks, and answer questions using only the uploaded document context.

## Project overview

The application loads PDF and TXT files, splits them into chunks, converts each chunk into embeddings using a sentence-transformer model, stores the vectors in ChromaDB, and then uses Gemini to answer user questions grounded in the retrieved document snippets. The interface is implemented with Streamlit so users can upload documents and ask follow-up questions without writing code.

## RAG architecture and workflow

1. Upload documents in PDF or TXT format.
2. The document loader reads each file and preserves metadata such as source filename and PDF page numbers.
3. `RecursiveCharacterTextSplitter` breaks long documents into smaller sections.
4. The sentence-transformer embedding model creates vectors for each chunk.
5. ChromaDB stores the chunks and embeddings in a separate collection for each processing session.
6. A similarity search retrieves the chunks most relevant to the user question.
7. Gemini receives only the retrieved snippets as context.
8. The answer is displayed with source citations, and the system explicitly states when the uploaded documents do not contain enough information.

## Technologies used

- Python
- Streamlit
- LangChain
- LangChain Community document loaders
- FastEmbed embeddings via `langchain_community.embeddings.FastEmbedEmbeddings`
- ChromaDB
- Google Gemini via `langchain-google-genai`
- Pytest

## Installation instructions for Windows

1. Open PowerShell in the project folder.
2. Create a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install the dependencies:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
4. Copy `.env.example` to a real `.env` file and add your Gemini API key.

## Environment variable configuration

Create a `.env` file in the project root using the example values below:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.5-flash
```

If `GEMINI_API_KEY` is missing, the app will show an error before generating an answer.

## How to run the Streamlit application

From the project folder:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Then open the local URL printed by Streamlit in the browser.

## Example document and question

Example document content:

```
Datheon is a technology company that focuses on data, AI, and automation.
The internship program trains students to work with Python, Generative AI,
vector databases, and large-language-model workflows.
```

Example question:

```
What technologies does Datheon focus on?
```

Example answer with citation:

```
Datheon focuses on data, AI, and automation, along with Python and Generative AI workflows.
Source: sample_documents/example.txt
```

For PDF files, the citation should include the page number, for example:

```
Source: example.pdf (Page 3)
```

## How to run tests

```powershell
.\.venv\Scripts\Activate.ps1
pytest
```

## Project limitations

- The app depends on internet access for the Gemini API call and the HuggingFace embedding model download the first time it is used.
- ChromaDB stores vector data locally on disk in the temporary directory for each processing session.
- Large documents may require more memory and processing time depending on your computer.
- The application is designed for document-grounded answers only; it intentionally refuses to answer questions outside the uploaded content.
