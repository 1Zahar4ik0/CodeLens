import sys
import os
import ast
import json
import argparse
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
import chromadb

parser = argparse.ArgumentParser()
parser.add_argument("base_dir")
parser.add_argument("--reindex", action="store_true")
args = parser.parse_args()

base_dir = args.base_dir
reindex = args.reindex

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
        print(f"ошибка по пути: {path_file}")

def _make_chunk(node, source, relative_path, class_name=None):
    source_code = ast.get_source_segment(source, node)
    if source_code is None:
        return None

    if class_name:
        qualified_name = f"{class_name}.{node.name}"
    else:
        qualified_name = node.name

    chunk_id = f"{relative_path}:{qualified_name}:{node.lineno}"

    return {
        "chunk_id": chunk_id,
        "name": qualified_name,
        "type": "class" if isinstance(node, ast.ClassDef) else "function",
        "file_path": relative_path,
        "start_line": node.lineno,
        "end_line": node.end_lineno,
        "docstring": ast.get_docstring(node) or "",
        "source_code": source_code,
    }

def _collect_from_body(body, source, relative_path, class_name=None):
    chunks = []
    for node in body:
        if isinstance(node, ast.ClassDef):
            chunk = _make_chunk(node, source, relative_path)
            if chunk:
                chunks.append(chunk)
            chunks.extend(
                _collect_from_body(node.body, source, relative_path, class_name=node.name)
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunk = _make_chunk(node, source, relative_path, class_name)
            if chunk:
                chunks.append(chunk)
    return chunks

def creating_chunks(dir):
    files_path = header_counter(dir)

    all_chunks = []

    for file_path in tqdm(files_path, desc="Индексирование файлов:"):
        source = read_file(file_path)
        if source is None:
            continue

        tree = parse_file(source, file_path)
        if tree is None:
            continue

        relative_path = os.path.relpath(file_path, dir)
        relative_path = relative_path.replace(os.sep, "/")
        all_chunks.extend(_collect_from_body(tree.body, source, relative_path))

    return all_chunks

def build_embedding_text(chunk):
    parts = [chunk["file_path"], chunk["name"]]
    if chunk["docstring"]:
        parts.append(chunk["docstring"])
    parts.append(chunk["source_code"])
    return "\n".join(parts)

chunks = creating_chunks(base_dir)

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
texts = [build_embedding_text(chunk) for chunk in chunks]
output = model.encode(
    texts,
    batch_size=12,
    max_length=1024,
    return_dense=True,
    return_sparse=True,
)
embeddings = output["dense_vecs"]
sparse_vecs = output["lexical_weights"]

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
    ids=[chunk["chunk_id"] for chunk in chunks],
    embeddings=embeddings.tolist(),
    documents=[chunk["source_code"] for chunk in chunks],
    metadatas=[{
        "name": chunk["name"],
        "type": chunk["type"],
        "file_path": chunk["file_path"],
        "start_line": chunk["start_line"],
        "end_line": chunk["end_line"],
        "docstring": chunk["docstring"],
    } for chunk in chunks],
)

sparse_path = os.path.join("./chroma_db", "sparse_vectors.json")
sparse_map = {
    chunk["chunk_id"]: {k: float(v) for k, v in sv.items()}
    for chunk, sv in zip(chunks, sparse_vecs)
}
with open(sparse_path, "w", encoding="utf-8") as f:
    json.dump(sparse_map, f)