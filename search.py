import os
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from scipy.special import expit
import torch
import time

@st.cache_resource
def load_models():
    embedder = SentenceTransformer("BAAI/bge-m3")

    reranker = CrossEncoder(
        "BAAI/bge-reranker-v2-m3",  # cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
        max_length=256,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    return embedder, reranker


embedder, reranker = load_models()

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("codelens")

_all = collection.get(include=["documents", "metadatas"])
_all_ids = _all["ids"]
_all_docs = _all["documents"]
_all_metas = _all["metadatas"]

_tokenized = [doc.lower().split() for doc in _all_docs]
_bm25 = BM25Okapi(_tokenized)


def search(query: str, top_k: int = 5) -> list[dict]:
    candidate_k = 15

    query_vector = embedder.encode([query])[0]
    vec_results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )

    bm25_scores = _bm25.get_scores(query.lower().split())
    bm25_top_indices = bm25_scores.argsort()[::-1][:candidate_k]

    candidate_ids = set(vec_results["ids"][0])
    for idx in bm25_top_indices:
        candidate_ids.add(_all_ids[idx])
    candidate_ids = list(candidate_ids)

    candidate_docs = []
    candidate_metas = []
    for cid in candidate_ids:
        idx = _all_ids.index(cid)
        candidate_docs.append(_all_docs[idx])
        candidate_metas.append(_all_metas[idx])

    pairs = [[query, doc] for doc in candidate_docs]
    logits = reranker.predict(pairs)  # Сырые логиты

    probabilities = expit(logits)

    # ШАГ 3: Сборка и сортировка результатов
    combined = []
    for i in range(len(candidate_ids)):
        meta = candidate_metas[i]
        combined.append({
            "chunk_id":    candidate_ids[i],
            "source_code": candidate_docs[i],
            "name":        meta.get("name", "Unknown"),
            "type":        meta.get("type", "Unknown"),
            "file_path":   meta.get("file_path", "Unknown"),
            "start_line":  meta.get("start_line", 0),
            "end_line":    meta.get("end_line", 0),
            "docstring":   meta.get("docstring", ""),
            "relevance":   round(float(probabilities[i]) * 100, 1)
        })

    # Сортируем по убыванию вероятности
    combined.sort(key=lambda x: x["relevance"], reverse=True)

    return combined[:top_k]


if __name__ == "__main__":
    test_queries = [
        "как создаётся токен доступа",
        "how does JWT verification work",
        "где проверяется суперпользователь"
    ]

    for q in test_queries:
        start = time.time()
        results = search(q, top_k=3)
        latency = time.time() - start

        print(f"\n⏱️ Latency: {latency:.3f} сек. | Запрос: '{q}'")
        for i, r in enumerate(results, 1):
            print(
                f"  {i}. [{r['relevance']:.1f}%] {r['file_path']} → {r['name']}")
