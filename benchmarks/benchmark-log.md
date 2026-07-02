# RAG Performance Benchmarks

This file tracks how the RAG system performs as the indexed collection grows.

| Date | PDFs | Chunks | Avg Query | Fastest | Slowest | Search Only | Notes |
|------|-----:|-------:|----------:|---------:|---------:|------------:|------|
| 2026-07-01 | 2 | 18 | 1467 ms | 1128 ms | 1960 ms | 302 ms | Initial Qdrant Cloud deployment |
| 2026-07-02 | ? | 173 | 2460 ms | 1301 ms | 3195 ms | 315 ms | Added CS5800 course materials |