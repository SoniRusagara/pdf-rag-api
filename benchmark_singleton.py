import time 
from vector_db import QdrantStorage 
"""
Benchmark: compares creating a new Qdrant connection on every call (old pattern) 
vs. reusing one shared connection (singleton, new pattern).
Result: shared connection was ~90% faster on average, across n=20 to 500 runs.
Used this to get a real, defensible number for my resume singleton refactor bullet.
"""

def embed_test_vector():
    # Using a fixed dummy vector to measure only connection overhead and not OpenAI embedding time 
    return [0.1] * 3072

def time_new_instance_per_call(n=500):
    vec = embed_test_vector()
    start = time.perf_counter()
    for _ in range(n):
        # This is a new connection every time, which was the old pattern
        store = QdrantStorage()
        store.search(vec, top_k=5)
    elapsed = time.perf_counter() - start
    return elapsed / n # average per call 

def time_shared_instance(n=500):
    vec = embed_test_vector()
    # Connect once, which is the new pattern
    store = QdrantStorage()
    start = time.perf_counter()
    for _ in range(n):
        store.search(vec, top_k=5)
    elapsed = time.perf_counter() - start
    return elapsed / n

if __name__ == "__main__":
    old_avg = time_new_instance_per_call()
    new_avg = time_shared_instance()
    improvement = (1 - (new_avg / old_avg)) * 100

    print("========================================")
    print(f"  New instance per call : {old_avg*1000:.1f}ms avg")
    print(f"  Shared instance       : {new_avg*1000:.1f}ms avg")
    print(f"  Improvement           : {improvement:.1f}%")
    print("========================================")