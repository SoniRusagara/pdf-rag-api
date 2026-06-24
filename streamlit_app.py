import asyncio
from pathlib import Path
import time

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(
    page_title="pdf-rag-api",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Hide Streamlit watermark and top menu */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header { visibility: hidden; }

  /* Sidebar dark background */
  [data-testid="stSidebar"] {
    background-color: #0e0e1a !important;
  }
  [data-testid="stSidebar"] * {
    color: #c0c0d8 !important;
  }
  [data-testid="stSidebar"] .st-emotion-cache-1cypcdb {
    background-color: #0e0e1a !important;
  }

  /* Radio nav items in sidebar */
  [data-testid="stSidebar"] label {
    color: #c0c0d8 !important;
    font-size: 14px !important;
  }

  /* Upload zone */
  [data-testid="stFileUploader"] {
    border: 1.5px dashed #AFA9EC;
    border-radius: 8px;
    background: #f0effe;
    padding: 8px;
  }

  /* Primary button */
  .stButton > button {
    background-color: #534AB7 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    width: 100%;
    padding: 10px !important;
  }
  .stButton > button:hover {
    background-color: #3C3489 !important;
  }

  /* Answer block */
  .answer-block {
    background: #f0effe;
    border-left: 4px solid #534AB7;
    border-radius: 0 8px 8px 0;
    padding: 14px 16px;
    margin-top: 12px;
    font-size: 15px;
    color: #1a1a2e;
    line-height: 1.6;
  }

  /* Source pills */
  .source-pill {
    display: inline-block;
    background: #EEEDFE;
    color: #3C3489;
    border: 1px solid #AFA9EC;
    border-radius: 99px;
    padding: 3px 10px;
    font-size: 12px;
    margin-right: 6px;
    margin-top: 6px;
  }

  /* Metric cards */
  .metric-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 10px;
    margin-top: 16px;
  }
  .metric-card {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 12px 14px;
  }
  .metric-val {
    font-size: 20px;
    font-weight: 700;
    color: white;
  }
  .metric-label {
    font-size: 11px;
    color: #9090b8;
    margin-top: 2px;
  }

  /* Status dot */
  .status-connected {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: #5DCAA5;
    margin-top: 4px;
  }
  .status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #1D9E75;
    display: inline-block;
  }

  /* Section headers */
  .section-title {
    font-size: 15px;
    font-weight: 600;
    color: #1a1a2e;
    margin-bottom: 2px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .section-sub {
    font-size: 12px;
    color: #888;
    margin-bottom: 14px;
  }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_path.write_bytes(file.getbuffer())
    return file_path


async def send_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


async def send_query_event(question: str, top_k: int):
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={"question": question, "top_k": top_k},
        )
    )
    return result[0]


def inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json().get("data", [])


def wait_for_output(event_id: str, timeout_s: float = 120.0) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out (last status: {last_status})")
        time.sleep(0.5)


def check_qdrant() -> bool:
    try:
        r = requests.get("http://localhost:6333/collections", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def get_doc_count() -> int:
    try:
        r = requests.get("http://localhost:6333/collections/docs", timeout=2)
        if r.status_code == 200:
            return r.json().get("result", {}).get("points_count", 0)
    except Exception:
        pass
    return 0


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 pdf-rag-api")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Home", "Ingest PDF", "Ask a question"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    qdrant_ok = check_qdrant()
    if qdrant_ok:
        st.markdown('<div class="status-connected"><span class="status-dot"></span> Qdrant connected</div>', unsafe_allow_html=True)
    else:
        st.error("Qdrant offline")

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("FastAPI · Inngest · Qdrant · OpenAI")


# ── Hero banner ─────────────────────────────────────────
doc_count = get_doc_count()

st.markdown(f"""
<div style="background: linear-gradient(135deg, #13103a 0%, #1e1560 100%);
            border-radius: 12px; padding: 28px 32px; margin-bottom: 24px;">
  <div style="display:inline-flex; align-items:center; gap:5px; font-size:11px;
              font-weight:600; padding:3px 10px; border-radius:99px;
              background:rgba(255,255,255,0.1); color:#c8c4f8;
              border:1px solid rgba(255,255,255,0.15); margin-bottom:12px;">
    ✦ AI-POWERED RAG PIPELINE
  </div>
  <h1 style="font-size:26px; font-weight:700; color:white; margin:0 0 6px; line-height:1.3;">
    Document intelligence<br>
    <span style="color:#9F97F0;">for your PDFs</span>
  </h1>
  <p style="font-size:14px; color:#a0a0c0; margin:0 0 20px; line-height:1.6; max-width:520px;">
    Upload any PDF. Ask questions. Get answers grounded in your actual documents
    using semantic vector search and GPT-4o-mini.
  </p>
  <div class="metric-row">
    <div class="metric-card">
      <div class="metric-val">{doc_count}</div>
      <div class="metric-label">chunks ingested</div>
    </div>
    <div class="metric-card">
      <div class="metric-val">3,072</div>
      <div class="metric-label">embedding dims</div>
    </div>
    <div class="metric-card">
      <div class="metric-val">gpt-4o-mini</div>
      <div class="metric-label">LLM model</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Pages ─────────────────────────────────────────────────────────────────────

if page == "Home":
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">✦ What this does</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Production-grade features under the hood</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Semantic search** — finds relevant chunks using vector similarity, not keyword matching
        - **Fault-tolerant ingestion** — Inngest retries failed steps automatically up to 5x
        - **Rate limiting** — prevents duplicate ingestion of the same source
        - **Observability** — every run is inspectable in the Inngest Dev Server UI
        - **Grounded answers** — LLM only uses retrieved context, no hallucination
        """)

    with col2:
        st.markdown('<div class="section-title">⚡ Tech stack</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Built for production, not just demos</div>', unsafe_allow_html=True)
        st.markdown("""
        | Layer | Tool |
        |---|---|
        | API server | FastAPI + Uvicorn |
        | Orchestration | Inngest |
        | Vector DB | Qdrant |
        | Embeddings | OpenAI text-embedding-3-large |
        | LLM | GPT-4o-mini |
        | PDF parsing | LlamaIndex |
        """)


elif page == "Ingest PDF":
    st.markdown('<div class="section-title">📤 Ingest a document</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload a PDF to chunk, embed, and store in Qdrant</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded:
        st.caption(f"Selected: **{uploaded.name}** ({round(uploaded.size / 1024, 1)} KB)")
        if st.button("Trigger ingestion"):
            with st.spinner("Saving and triggering Inngest workflow..."):
                path = save_uploaded_pdf(uploaded)
                asyncio.run(send_ingest_event(path))
                time.sleep(0.3)
            st.success(f"Ingestion triggered for **{uploaded.name}**")
            st.caption("Monitor progress in the Inngest Dev Server at http://localhost:8288")


elif page == "Ask a question":
    st.markdown('<div class="section-title">💬 Ask a question</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Retrieve relevant context and generate a grounded answer</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input(
            "Your question",
            placeholder="What are the key skills listed in this document?",
            label_visibility="collapsed",
        )
    with col2:
        top_k = st.number_input("Chunks", min_value=1, max_value=20, value=5, step=1)

    if st.button("Generate answer") and question.strip():
        with st.spinner("Searching documents and generating answer..."):
            event_id = asyncio.run(send_query_event(question.strip(), int(top_k)))
            output = wait_for_output(event_id)

        answer = output.get("answer", "")
        sources = output.get("sources", [])
        num_contexts = output.get("num_contexts", 0)

        if answer:
            pills = "".join(
                f'<span class="source-pill">📄 {s}</span>' for s in sources
            )
            st.markdown(f"""
            <div class="answer-block">
              {answer}
              <div style="margin-top:10px; border-top:1px solid #d0c8f8; padding-top:8px;">
                <span style="font-size:11px; color:#888;">Sources ({num_contexts} chunks retrieved)</span><br>
                {pills}
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No answer returned. Make sure you have ingested a PDF first.")