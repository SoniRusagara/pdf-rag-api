"""
Retrieval accuracy evaluation
Measures whether search() returns the correct source document for known questions
"""

from data_loader import embed_texts
from vector_db import QdrantStorage

EVAL_SET = [
    {"question": "Why can traditional retrieval metrics like precision and recall be misleading when evaluating a retriever inside a full question-answering system?", "expected_source": "retrieval-eval-q1"},
    {"question": "How did the Boston Gazette and Boston News-Letter differ in their political perspectives?", "expected_source": "retrieval-eval-qBoston-newspapers"},
    {"question": "How did FIFA's medical initiatives at the 2006 World Cup extend beyond simply tracking player injuries?", "expected_source": "retrieval-eval-qWorldCup"},
    {"question": "Which of the three major hypotheses about belief in astrology was not supported by the 1988 British survey data?", "expected_source": "retrieval-eval-qAstrology"},
    {"question": "What role did offensive and defensive statistics play in predicting NBA team performance according to the study?", "expected_source": "retrieval-eval-qNBA"},
    {"question": "What were the three most common categories of illness among ill returned European travellers in 2008?", "expected_source": "retrieval-eval-qtravel"},
    {"question": "How many Important Bird Areas are recognized in Cape Verde?", "expected_source": "retrieval-eval-qCapeVerde"},
    {"question": "What evidence suggested that wild and domestic cats are genetically distinct populations?", "expected_source": "retrieval-eval-qCats"},
    {"question": "What was the main objective of the 2014 Annals of Family Medicine trial on therapeutic massage for chronic neck pain?", "expected_source": "retrieval-eval-qMassage-obj"},
    {"question": "What is the difference between the 'True Bimbo' and the 'Recontextualized Bimbo'?", "expected_source": "bimbo-eval-q1"},

]

def run_eval():
    store = QdrantStorage()
    top1_correct = 0
    top3_correct = 0 
    results_log = []

    for item in EVAL_SET:
        question = item["question"]
        expected = item["expected_source"]

        query_vec = embed_texts([question])[0]
        result = store.search(query_vec, top_k = 3)
        retrieved_sources = result["sources"]

        # Since sources is deduplicated by your search() function, we check
        # membership rather than exact position for top-3
        is_top3 = expected in retrieved_sources
        is_top1 = len(retrieved_sources) > 0 and retrieved_sources[0] == expected

        if is_top1:
            top1_correct += 1
        if is_top3:
            top3_correct += 1
        
        results_log.append({
            "question": question,
            "expected": expected,
            "retrieved": retrieved_sources,
            "top1_hit": is_top1,
            "top3_hit": is_top3,
        })

    total = len(EVAL_SET)
    print(f"\n{'='*50}")
    print(f"  Retrieval Accuracy Evaluation")
    print(f"{'='*50}")
    print(f"  Total questions   : {total}")
    print(f"  Top-1 accuracy    : {top1_correct}/{total} ({100*top1_correct/total:.1f}%)")
    print(f"  Top-3 accuracy    : {top3_correct}/{total} ({100*top3_correct/total:.1f}%)")
    print(f"{'='*50}\n")

    # Show any misses for debugging
    misses = [r for r in results_log if not r["top3_hit"]]
    if misses:
        print("Missed questions:")
        for m in misses:
            print(f"  Q: {m['question']}")
            print(f"     Expected: {m['expected']}, Got: {m['retrieved']}")

if __name__ == "__main__":
    run_eval()

