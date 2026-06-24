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
