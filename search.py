import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

model = SentenceTransformer("BAAI/bge-m3")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("codelens")

_all = collection.get(include=["documents", "metadatas"])
_all_ids = _all["ids"]
_all_docs = _all["documents"]
_all_metas = _all["metadatas"]

_tokenized = [doc.lower().split() for doc in _all_docs]
_bm25 = BM25Okapi(_tokenized)


def _normalize(scores: list[float]) -> list[float]:
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def search(query: str, top_k: int = 5, alpha: float = 0.7) -> list[dict]:
    total = collection.count()

    query_vector = model.encode([query])[0]
    vector_results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=total,
        include=["documents", "metadatas", "distances"],
    )

    vec_ids = vector_results["ids"][0]
    vec_distances = vector_results["distances"][0]
    vec_metadatas = vector_results["metadatas"][0]
    vec_documents = vector_results["documents"][0]

    vec_scores = [1 - d / 2 for d in vec_distances]
    vec_scores_norm = _normalize(vec_scores)

    tokenized_query = query.lower().split()
    bm25_scores = _bm25.get_scores(tokenized_query).tolist()
    bm25_norm = _normalize(bm25_scores)

    id_to_bm25 = {_all_ids[i]: bm25_norm[i] for i in range(len(_all_ids))}

    combined = []
    for i, chunk_id in enumerate(vec_ids):
        bm25_score = id_to_bm25.get(chunk_id, 0.0)
        final_score = alpha * vec_scores_norm[i] + (1 - alpha) * bm25_score

        meta = vec_metadatas[i]
        combined.append({
            "chunk_id": chunk_id,
            "source_code": vec_documents[i],
            "name": meta["name"],
            "type": meta["type"],
            "file_path": meta["file_path"],
            "start_line": meta["start_line"],
            "end_line": meta["end_line"],
            "docstring": meta["docstring"],
            "relevance": round(final_score * 100, 1),
        })

    combined.sort(key=lambda x: x["relevance"], reverse=True)
    return combined[:top_k]


if __name__ == "__main__":
    query = "как создаётся токен доступа"
    results = search(query, top_k=5)

    print(f"Запрос: {query}\n")
    for r in results:
        print(
            f"[{r['relevance']}%] {r['file_path']} → {r['type']} {r['name']} (строки {r['start_line']}–{r['end_line']})")
