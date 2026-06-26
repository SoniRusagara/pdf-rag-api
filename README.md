# 📄 PDF RAG API

A Retrieval-Augmented Generation (RAG) pipeline for ingesting and querying PDF documents using semantic search and LLM-powered responses. Built to explore how production-oriented AI systems handle document ingestion, retrieval, and workflow orchestration beyond a simple project.

Unlike most RAG demos, this project focuses on operational concerns such as workflow orchestration, fault tolerance, observability, and ingestion controls.

## Overview

Upload a PDF, ask a question, and receive a grounded answer based on the document's contents.

### Ingestion Flow

PDF → Parse & Chunk → OpenAI Embeddings → Qdrant

### Query Flow

Question → Embedding → Similarity Search → GPT-4o-mini → Answer

## Tech Stack

| Layer                  | Tool                            |
| ---------------------- | ------------------------------- |
| API Server             | FastAPI + Uvicorn               |
| Workflow Orchestration | Inngest                         |
| Vector Database        | Qdrant                          |
| Embeddings             | OpenAI `text-embedding-3-large` |
| LLM                    | OpenAI `gpt-4o-mini`            |
| PDF Processing         | LlamaIndex                      |
| Frontend               | Streamlit                       |
| Package Management     | uv                              |

## ⚙️ Production Features

This project includes several patterns commonly used in production AI systems:

* **Automatic retries**: failed workflow steps are retried up to 5 times to handle transient API failures
* **Rate limiting**: limits ingestion frequency to prevent duplicate processing and unnecessary API costs
* **Throttling**: controls workflow execution volume during ingestion
* **Step-level observability**: inspect logs, execution times, and workflow state through the Inngest dashboard
* **Replayable workflows**: rerun failed or completed workflows for debugging and testing
* **Event-driven architecture**: ingestion and querying run as asynchronous workflows rather than blocking API requests

## Local Setup

### Prerequisites

* Python 3.12+
* uv
* Docker
* Node.js
* OpenAI API key

### 1. Clone and Install

```bash
git clone https://github.com/sonirusagara/pdf-rag-api.git

cd pdf-rag-api

uv venv
uv sync
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Add your OpenAI API key:

```env
OPENAI_API_KEY=sk-...
```

### 3. Start Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 4. Start the FastAPI Server

```bash
uv run uvicorn main:app
```

### 5. Start the Inngest Dev Server

```bash
npx inngest-cli@latest dev
```

### 6. Start the Streamlit Frontend

```bash
uv run streamlit run streamlit_app.py
```

Open:

* `http://localhost:8501` — Streamlit UI
* `http://localhost:8288` — Inngest Dashboard

## Triggering Workflows Manually

### Ingest a PDF

```json
{
  "name": "rag/ingest_pdf",
  "data": {
    "pdf_path": "/absolute/path/to/file.pdf"
  }
}
```

### Query the Knowledge Base

```json
{
  "name": "rag/query_pdf_ai",
  "data": {
    "question": "What are the main skills listed in this document?"
  }
}
```

## License

MIT
