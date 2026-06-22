import json
import streamlit as st
import chromadb
from FlagEmbedding import BGEM3FlagModel


@st.cache_resource
def _load_model():
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    reranker = CrossEncoder(
        "BAAI/bge-reranker-v2-m3",  # cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
        max_length=256,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    return embedder, reranker


embedder, reranker = load_models()

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("codelens")

with open("./chroma_db/sparse_vectors.json", "r", encoding="utf-8") as f:
    _sparse_index = json.load(f)

def _normalize(scores: list[float]) -> list[float]:
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def _is_russian(text: str) -> bool:

    if not text:
        return False
    russian = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    return russian / len(text) > 0.2


def _auto_alpha(query: str) -> float:

    return 0.9 if _is_russian(query) else 0.5


def search(query: str, top_k: int = 5, alpha: float = None) -> list[dict]:

    if alpha is None:
        alpha = _auto_alpha(query)

    n_candidates = min(max(top_k * 6, 30), collection.count())

    out = model.encode([query], return_dense=True, return_sparse=True)
    query_vector = out["dense_vecs"][0]
    query_sparse = out["lexical_weights"][0]

    vector_results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=n_candidates,
        include=["documents", "metadatas", "distances"],
    )

    vec_ids       = vector_results["ids"][0]
    vec_distances = vector_results["distances"][0]
    vec_metadatas = vector_results["metadatas"][0]
    vec_documents = vector_results["documents"][0]

    vec_scores      = [1 - d / 2 for d in vec_distances]
    vec_scores_norm = _normalize(vec_scores)

    sparse_scores = [
        model.compute_lexical_matching_score(
            query_sparse, _sparse_index.get(chunk_id, {})
        )
        for chunk_id in vec_ids
    ]
    sparse_scores_norm = _normalize(sparse_scores)

    combined = []
    for i, chunk_id in enumerate(vec_ids):
        final_score = alpha * vec_scores_norm[i] + (1 - alpha) * sparse_scores_norm[i]

        meta = vec_metadatas[i]
        combined.append({
            "chunk_id":    chunk_id,
            "source_code": vec_documents[i],
            "name":        meta["name"],
            "type":        meta["type"],
            "file_path":   meta["file_path"],
            "start_line":  meta["start_line"],
            "end_line":    meta["end_line"],
            "docstring":   meta["docstring"],
            "relevance":   round(final_score * 100, 1),
        })

    # Сортируем по убыванию вероятности
    combined.sort(key=lambda x: x["relevance"], reverse=True)

    return combined[:top_k]


if __name__ == "__main__":
    for query in ["как создаётся токен доступа", "how does JWT verification work"]:
        alpha = _auto_alpha(query)
        print(f"\nЗапрос: {query}  (alpha={alpha})")
        for r in search(query, top_k=3):
            print(f"  [{r['relevance']}%] {r['file_path']} -> {r['name']}")