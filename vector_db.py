import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantStorage:
    def __init__(self, url=None, api_key=None, collection="docs", dim=3072):
        # Falls back to env vars, then to local defaults if nothing is set 
        url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = api_key or os.getenv("QDRANT_API_KEY")
        
        # Crashes if db doesnt connect in 30seconds 
        self.client = QdrantClient(url=url, api_key=api_key, timeout=30)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection, 
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    # Fx converts text chunks into Qdrant-compatible points (vector + metadata)
    # and inserts or updates them in the vector db via creating a PointStructure
    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    # Fx to search for vectors 
    def search(self, query_vector, top_k: int = 5):
        # Finding the most similar stored chunks to the question 
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector, 
            with_payload=True,
            limit=top_k
        ).points
        # Stores the retrieved text chunks to be used as context for the AI prompt
        contexts = [] 
        # Stores the identifying information or file names for where the retrieved chunks originated
        sources = set() # No duplicates 

        # Looping through search results 
        for r in results:
            payload = getattr(r, "payload", None) or {} # Empty dict. if nothing exists
            text = payload.get("text", "") # actual content 
            source = payload.get("source", "") # file name / origin 
            # Building list of useful text chunks & list of sources 
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}
