import sys
import os
import ast
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

if len(sys.argv) > 1:
    base_dir = sys.argv[1]
else:
    print("Ошибка")
    sys.exit(1)

if not os.path.isdir(base_dir):
    print(f"Ошибка, это не директория: {base_dir}")
    sys.exit(1)


def header_counter(base_dir):
    arr_files = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.py'):
                path_file = os.path.join(root, file)
                arr_files.append(path_file)

    print(f"Всего найдено {len(arr_files)} py файлов")
    return arr_files


def read_file(path):
    try:
        with open(path, 'r', encoding="utf-8") as f:
            return f.read()
    except (UnicodeDecodeError, OSError):
        return None


def parse_file(code, path_file):
    try:
        tree = ast.parse(code)
        return tree
    except SyntaxError:
        print(f"ошибка в по пути: {path_file}")


TARGET_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def creating_chunks(dir):
    files_path = header_counter(dir)

    all_chunks = []

    for file_path in tqdm(files_path, desc="Индексирование файлов"):
        source = read_file(file_path)
        if source is None:
            continue

        tree = parse_file(source, file_path)
        if tree is None:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, TARGET_NODES):
                continue

            source_code = ast.get_source_segment(source, node)
            if source_code is None:
                continue

            relative_path = os.path.relpath(file_path, dir)
            chunk_id = f"{relative_path}::{node.name}::{node.lineno}"

            all_chunks.append({
                "chunk_id": chunk_id,
                "name": node.name,
                "type": "class" if isinstance(node, ast.ClassDef) else "function",
                "file_path": relative_path,
                "start_line": node.lineno,
                "end_line": node.end_lineno,
                "docstring": ast.get_docstring(node) or "",
                "source_code": source_code,
            })

    return all_chunks


def build_embedding_text(chunk):
    parts = [chunk["name"]]
    if chunk["docstring"]:
        parts.append(chunk["docstring"])
    parts.append(chunk["source_code"])
    return "\n".join(parts)


chunks = creating_chunks(base_dir)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
texts = [build_embedding_text(c) for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("codelens")

existing = collection.count()
if existing > 0:
    if not reindex:
        print(f"Всего {existing} чанков. Используй --reindex для перезаписи.")
        sys.exit(0)
    else:
        print(f"Удаляем старый индекс ({existing} чанков)...")
        client.delete_collection("codelens")
        collection = client.get_or_create_collection("codelens")

collection.add(
    ids=[c["chunk_id"] for c in chunks],
    embeddings=embeddings.tolist(),
    documents=[c["source_code"] for c in chunks],
    metadatas=[{
        "name": c["name"],
        "type": c["type"],
        "file_path": c["file_path"],
        "start_line": c["start_line"],
        "end_line": c["end_line"],
        "docstring": c["docstring"],
    } for c in chunks],
)
