"""
Performance tests for pdf-rag-api
Measures latency against live Qdrant + OpenAI
"""

import time 
import pytest 
import requests 
from dotenv import load_dotenv

load_dotenv()

# ── helpers ───────────────────────────────────────────────────────────────────
def qdrant_is_running() -> bool:
    try:
        r = requests.get("http://localhost:6333/collections", timeout=2)
        return r.status_code == 200
    except Exception: 
        return False 

def collection_has_data() -> bool: 
    try: 
        r = requests.get("http://localhost:6333/collections/docs", timeout=2)
        count = r.json().get("result", {}).get("points_count", 0)
        return count > 0
    except Exception:
        return False


# ── fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module", autouse=True)
def require_live_services():
    """Skip all performance tests if Qdrant is not running or has no data"""
    if not qdrant_is_running():
        pytest.skip("Qdrant not running - start wiht: docker run -p 6333:6333 qdrant/qdrant")
    if not collection_has_data():
        pytest.skip("No data in Qdrant - ingest a PDF first via the Inngest Dev Server")

# ── performance tests ─────────────────────────────────────────────────────────
class TestQueryLatency:
    """ Measures end-to-end query latency (embedding + vector search)"""

    SAMPLE_QUESTIONS = [
        "What are the key skills listed in this document?",
        "What is the main topic of this PDF?",
        "Summarize the most important points.",
    ]

    def test_single_query_under_5_seconds(self):
        """Single query (embedding + search) should complete in under 5s"""
        #TODO: Make this much faster as it is a loose sanity check 
        from data_loader import embed_texts
        from vector_db import QdrantStorage 

        start = time.time()
        query_vec = embed_texts([self.SAMPLE_QUESTIONS[0]])[0]
        store = QdrantStorage()
        store.search(query_vec, top_k=5)
        elapsed_ms = (time.time() - start) * 1000

        print(f"\n Single query latency: {elapsed_ms:.0f}ms")
        assert elapsed_ms < 5000, f"Query took {elapsed_ms:.0f}ms, expected under 5000ms"