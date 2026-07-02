from vector_db import QdrantStorage

qdrant_store = QdrantStorage()

print("PDFs indexed:", qdrant_store.count_unique_sources())