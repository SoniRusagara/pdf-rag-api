# RAG Performance Benchmarks

This file tracks how the RAG system performs as the indexed collection grows.

| Date | PDFs | Chunks | Avg Query | Fastest | Slowest | Search Only | Notes |
|------|-----:|-------:|----------:|---------:|---------:|------------:|------|
| 2026-07-01 | 2 | 18 | 1467 ms | 1128 ms | 1960 ms | 302 ms | Initial Qdrant Cloud deployment |
| 2026-07-02 | 33 | 173 | 2460 ms | 1301 ms | 3195 ms | 315 ms | Added CS5800 course materials |
| 2026-07-03 | 62 | 518 | 1324 ms | 1204 ms | 1484 ms | 340 ms | Indexed additional PDFs (518 chunks total); maintained sub-1.4 s average query latency and sub-350 ms vector retrieval |