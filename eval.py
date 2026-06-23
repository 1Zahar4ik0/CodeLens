import json
import time
from search import search

def load_eval_data(filepath):
    with open(filepath, 'r', encoding="utf-8") as f:
        return json.load(f)

def evaluate_precision(eval_data, top_k: int = 5):

    results = []
    precision = float()
    total_prec = 0.0
    total_lat = 0.0

    for item in eval_data:
        question_id = item["question_id"]
        question = item["query"]
        true_chunk_ids = set(item["correct_chunk_ids"])
        difficulty = item.get("difficulty", "unknown")

        start_time = time.perf_counter()
        predictions = search(question, top_k=top_k)
        end_time = time.perf_counter()
        latency = end_time-start_time
        total_lat += latency
        predicted_chunk_ids = [p.get("chunk_id") for p in predictions]

        hits = len(set(predicted_chunk_ids).intersection(true_chunk_ids))
        precision = hits / top_k
        total_prec += precision

        results.append({
            "question_id": question_id,
            "query": question,
            "difficulty": difficulty,
            "true_ids": list(true_chunk_ids),
            "predicted_ids": predicted_chunk_ids,
            "hits": hits,
            "precision@5": precision,
            "latency": round(latency, 3)
        })

    avg_prec = total_prec / len(eval_data)
    avg_lat = total_lat / len(eval_data)

    return {"results": results,
            "avg_precision": avg_prec,
            "avg_latency": avg_lat}

def print_console_report(report):
    avg_prec = report["avg_precision@5"]
    avg_lat = report["avg_latency_sec"]

    prec_met = avg_prec >= 0.60
    lat_met = avg_lat <= 3.0

    print("=" * 100)
    print("ИТОГОВЫЙ ОТЧЕТ ПО КАЧЕСТВУ RAG-СИСТЕМЫ (CodeLens)")
    print("=" * 100)

    prec_status = "ПРОЙДЕНО (≥ 60%)" if prec_met else "НЕ ПРОЙДЕНО (< 60%)"
    lat_status = "ПРОЙДЕНО (≤ 3 сек)" if lat_met else "НЕ ПРОЙДЕНО (> 3 сек)"

    print(f"Precision@5 : {avg_prec:.2%}  |  {prec_status}")
    print(f"Avg Latency : {avg_lat:.3f} сек |  {lat_status}")
    print("=" * 100)
    print(f"{'ID':<6} | {'Сложн.':<7} | {'Hits':<4} | {'P@5':<5} | {'Время':<6} | {'Запрос (query)'}")
    print("-" * 100)

    for res in report["results"]:
        q_short = res["query"][:55] + \
            "..." if len(res["query"]) > 55 else res["query"]

        print(
            f"{res['question_id']:<6} | "
            f"{res['difficulty']:<7} | "
            f"{res['hits']:<4} | "
            f"{res['precision@5']:.2f}  | "
            f"{res['latency_sec']:<6.3f} | "
            f"{q_short}"
        )

    print("=" * 100)

if __name__ == "__main__":
    eval_data = "data/eval_questions.json"

    report = evaluate_precision(eval_data=eval_data)

    print_console_report(report=report)