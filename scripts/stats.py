import os
from dotenv import load_dotenv
from vector_db import QdrantStorage
"""
uv run python -m scripts.stats
"""

load_dotenv()

qdrant_store = QdrantStorage()

results, _ = qdrant_store.client.scroll(
    collection_name=qdrant_store.collection,
    limit=10000,
    with_payload=True,
    with_vectors=False,
)

sources = {
    os.path.basename(r.payload.get("source", ""))
    for r in results
}

print("PDFs indexed:", len(sources))
print("Sources:")
for source in sorted(sources):
    print(f" - {source}")
