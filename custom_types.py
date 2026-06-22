import pydantic 

# A Pydantic data model that groups text chunks 
# with their source file path for the RAG pipeline
class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: str = None 

# A Pydantic data model that tracks the 
# number of items successfully ingested 
class RAGUpsertResult(pydantic.BaseModel):
    ingested: int

# A Pydantic model for search results 
# containing retrieved text and source paths
class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]

# A Pydantic model for final query results 
# with AI answers and sources
class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int
