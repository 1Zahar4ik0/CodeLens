import json
import sys
from search import search


def load_eval_data(filepath):
    with open(filepath, 'r', encoding="utf-8") as f:
        return json.load(f)


def evaluate_precision(eval_data, top_k: int = 5):

    results = []
    precision = float()

    for i, item in enumerate(eval_data, 1):
        question_id = item.get("question_id", f"q_{i}")
        question = item["query"]

        true_chunk_ids = set(item["correct_chunk_ids"])
        difficulty = item.get("difficulty", "unknown")

        predictions = search(question, top_k=5)
        predicted_chunk_ids = [p.get("chunk_id") for p in predictions]

        hits = len(set(predicted_chunk_ids).intersection(true_chunk_ids))
        precision = hits / 5

        results.append({
            "question_id": question_id,
            "query": question,
            "difficulty": difficulty,
            "true_ids": list(true_chunk_ids),
            "predicted_ids": predicted_chunk_ids,
            "hits": hits,
            "precision@5": precision
        })