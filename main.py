from fastapi import FastAPI
import logging 
import inngest 
import inngest.fast_api 
from inngest.experimental import ai 
from dotenv import load_dotenv
import uuid 
import os 
import datetime 
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage 
from custom_types import RAGQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

# Single shared connection, resued across all requests
qdrant_store = QdrantStorage()

# Run this workflow when a PDF ingestion event is received
@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(
        count=2, period=datetime.timedelta(minutes=1)
    ),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
    ),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    # Internal step to extract and split PDF content for processing
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        # TODO: Add explaining comments here 
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    # Internal step to generate embeddings and 
    # save chunks to the vector database
    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        # Extract chunks and source information
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        # Convert text chunks into embedding vectors
        vecs = embed_texts(chunks)

        # Generate a unique ID for each chunk
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, name=f"{source_id}:{i}")) for i in range(len(chunks))]
        
        # Store source metadata alongside each chunk
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        
        # Save embeddings and metadata to Qdrant
        qdrant_store.upsert(ids, vecs, payloads)

        # Return the number of chunks ingested
        return RAGUpsertResult(ingested=len(chunks))

    # Load the PDF and split it into chunks
    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    # Generate embeddings and store chunks in Qdrant
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    # Return ingestion results as a dictionary
    return ingested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Query PDF", 
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        # Convert the user's question into an embedding vector 
        query_vec = embed_texts([question])[0]
        # Retrieve the most relevant chunks from Qdrant 
        found = qdrant_store.search(query_vec, top_k)
        # TODO: Add comment here 
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])
    
    # Get the user's question from the event payload 
    question = ctx.event.data["question"]
    #Number of relevant chunks to retrieve (default = 5)
    top_k = int(ctx.event.data.get("top_k", 5))

    # Generate a query embedding & retrieve matching chunks from Qdrant 
    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    # Combine retrieved chunks into a single context block for the LLM
    context_block = "\n\n".join(f"- {c}" for c in found.contexts)

    # Create the prompt using retrieved context and the user's question 
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    # Generate an answer using the retrieved context
    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}

            ]
        }
    )

    # Return the answer along with source information and retrieval stats
    answer = res["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}

app = FastAPI()


inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])