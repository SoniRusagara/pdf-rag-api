# RAG Performance Benchmarks

This file tracks how the RAG system performs as the indexed collection grows.

| Date | PDFs | Chunks | Avg Query | Fastest | Slowest | Search Only | Notes |
|------|-----:|-------:|----------:|---------:|---------:|------------:|------|
| 2026-07-01 | 2 | 18 | 1467 ms | 1128 ms | 1960 ms | 302 ms | Initial Qdrant Cloud deployment |
| 2026-07-02 | 33 | 173 | 2460 ms | 1301 ms | 3195 ms | 315 ms | Added CS5800 course materials |
| 2026-07-03 | 62 | 518 | 1324 ms | 1204 ms | 1484 ms | 340 ms | Indexed additional PDFs (518 chunks total); maintained sub-1.4 s average query latency and sub-350 ms vector retrieval |
| 2026-07-03 | 95 | 1085 | 1399 ms | 1313 ms | 1496 ms | 333 ms | Expanded corpus to 1,085 chunks and added concurrent query benchmark (6 concurrent queries, ~1.94 s total wall time); 5/5 performance tests passed. |
| 2026-07-03 | 156 | 2253 | 1293 ms | 1259 ms | 1321 ms | 322 ms | Expanded corpus to 2,253 chunks across 95 PDFs; added concurrent query benchmark (6 concurrent queries, 1.51 s total wall time); 5/5 performance tests passed. |