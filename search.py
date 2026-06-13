import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("codelens")

def search(query, top_k: int = 5):
    query_vector = model.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=top_k,
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    output = []
    for i in range(len(ids)):
        relevance = round((1 - distances[i] / 2) * 100, 1)
        output.append({
            "chunk_id": ids[i],
            "source_code": documents[i],
            "name": metadatas[i]["name"],
            "type": metadatas[i]["type"],
            "file_path": metadatas[i]["file_path"],
            "start_line": metadatas[i]["start_line"],
            "end_line": metadatas[i]["end_line"],
            "docstring": metadatas[i]["docstring"],
            "relevance": relevance,
        })

    return output


if __name__ == "__main__":
    query = "как обрабатываются ошибки авторизации"
    results = search(query, top_k=5)

    print(f"Запрос: {query}\n")
    for r in results:
        print(f"[{r['relevance']}%] {r['file_path']} → {r['type']} {r['name']} (строки {r['start_line']}–{r['end_line']})")