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

    def test_average_latency_across_3_runs(self):
        """Runs 3 queries & prints average latency"""
        from data_loader import embed_texts
        from vector_db import QdrantStorage

        latencies = []

        for i, question in enumerate(self.SAMPLE_QUESTIONS):
            start = time.time()
            query_vec = embed_texts([question])[0]
            store = QdrantStorage()
            store.search(query_vec, top_k=5)
            elapsed_ms = (time.time() - start) * 1000
            latencies.append(elapsed_ms)
            print(f"\n Run {i+1}: {elapsed_ms:.0f}ms - '{question[:40]}...'")

        avg = sum(latencies) / len(latencies)
        fastest = min(latencies)
        slowest = max(latencies)

        print(f"\n {'='*40}")
        print(f"  Average latency : {avg:.0f}ms")
        print(f"  Fastest         : {fastest:.0f}ms")
        print(f"  Slowest         : {slowest:.0f}ms")
        print(f"  {'='*40}")
        print(f"  Resume Point    : <{round(avg/100)*100 + 100}ms avg query latency")

        # Should average under 10 seconds (network to OpenAI + Qdrant search)
        assert avg < 10000, f"Average latency {avg:.0f}ms is too slow"

    
    def test_qdrant_search_only_latency(self):
        """
        Measures the Qdrant vector search without an OpenAI call. 
        To isolate db performance & network latency 
        """
        from data_loader import embed_texts
        from vector_db import QdrantStorage

        # Get embedding once (outside timed section)
        query_vec = embed_texts([self.SAMPLE_QUESTIONS[0]])[0]
        store = QdrantStorage()

        # Time the Qdrant search 
        start = time.time()
        results = store.search(query_vec, top_k=5)
        elapsed_ms = (time.time() - start) * 1000

        print(f"\n Qdrant search only: {elapsed_ms:.0f}ms")
        print(f"  Chunks retrieved  : {len(results['contexts'])}")
        assert elapsed_ms < 1000, f"Qdrant search took {elapsed_ms:.0f}ms, expected under 1000ms"



class TestIngestionScale:
    """Reports on the scale of data currently in the vector DB"""

    def test_report_collection_stats(self):
        """Prints current collection size - TODO: use this for resume metrics"""
        r = requests.get("http://localhost:6333/collections/docs")
        data = r.json().get("result", {})
        points = data.get("points_count", 0)
        config = data.get("config", {})
        vector_size = config.get("params", {}).get("vectors", {}).get("size", "unknown")

        print(f"\n {'='*40}")
        print(f"  Chunks in Qdrant  : {points}")
        print(f"  Embedding dims    : {vector_size}")
        print(f"  {'='*40}")
        print(f"  TODO-PUT ON RESUME     : ingested {points} chunks across indexed PDFs")
        print(f"                      using {vector_size}-dimensional embeddings")

        assert points > 0, "No chunks found - ingest a PDF first"
        assert vector_size == 3072, f"Expected 3072-D embeddings, got {vector_size}"

