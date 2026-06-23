import sys
import os
import ast
import re
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
    py_count = 0
    js_count = 0

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.py'):
                arr_files.append(os.path.join(root, file))
                py_count += 1
            elif file.endswith('.js'):
                arr_files.append(os.path.join(root, file))
                js_count += 1

    print(f"Всего найдено файлов: {py_count} .py, {js_count} .js")
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


def parse_python_chunks(source, relative_path, file_path):
    tree = parse_file(source, file_path)
    if tree is None:
        return []
    return _collect_from_body(tree.body, source, relative_path)

_JS_PATTERNS = [
    re.compile(r'^(?:export\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{', re.MULTILINE),

    re.compile(r'^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(', re.MULTILINE),

    re.compile(r'^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?(?:\(.*?\)\s*=>|\bfunction\b)', re.MULTILINE),

    re.compile(r'^\s{2,}(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{', re.MULTILINE),
]


def _extract_block(source, start_pos):
    depth = 0
    i = start_pos

    while i < len(source):
        if source[i] == '{':
            depth += 1
        elif source[i] == '}':
            depth -= 1
            if depth == 0:
                block = source[start_pos:i + 1]
                end_line = source[:i].count('\n') + 1
                return block, end_line
        i += 1
    return source[start_pos:], source.count('\n') + 1


def parse_js_chunks(source, relative_path):
    chunks = []
    lines = source.splitlines()
    seen_positions = set()

    for pattern in _JS_PATTERNS:
        for match in pattern.finditer(source):
            name = match.group(1)
            if name in ('if', 'for', 'while', 'switch', 'catch', 'return'):
                continue

            start_line = source[:match.start()].count('\n') + 1

            brace_pos = source.find('{', match.start())
            if brace_pos == -1:
                continue
            if brace_pos in seen_positions:
                continue
            seen_positions.add(brace_pos)

            source_code, end_line = _extract_block(source, brace_pos)
            if end_line - start_line < 1:
                continue

            chunk_id = f"{relative_path}:{name}:{start_line}"

            chunks.append({
                "chunk_id":   chunk_id,
                "name":       name,
                "type":       "class" if "class" in match.group(0) else "function",
                "file_path":  relative_path,
                "start_line": start_line,
                "end_line":   end_line,
                "docstring":  "",
                "source_code": source_code[:2000],
            })

    return chunks

def creating_chunks(dir):
    files_path = header_counter(dir)
    all_chunks = []

    for file_path in tqdm(files_path, desc="Индексирование файлов"):
        source = read_file(file_path)
        if source is None:
            continue

        relative_path = os.path.relpath(file_path, dir).replace(os.sep, "/")

        if file_path.endswith('.py'):
            all_chunks.extend(parse_python_chunks(source, relative_path, file_path))

        elif file_path.endswith('.js'):
            all_chunks.extend(parse_js_chunks(source, relative_path))

    return all_chunks

def build_embedding_text(chunk):
    parts = [chunk["file_path"], chunk["name"]]
    if chunk["docstring"]:
        parts.append(chunk["docstring"])
    parts.append(chunk["source_code"])
    return "\n".join(parts)

chunks = creating_chunks(base_dir)

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
texts = [build_embedding_text(c) for c in chunks]
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
        "name":       chunk["name"],
        "type":       chunk["type"],
        "file_path":  chunk["file_path"],
        "start_line": chunk["start_line"],
        "end_line":   chunk["end_line"],
        "docstring":  chunk["docstring"],
    } for chunk in chunks],
)

sparse_path = os.path.join("./chroma_db", "sparse_vectors.json")
sparse_map = {
    c["chunk_id"]: {k: float(v) for k, v in sv.items()}
    for c, sv in zip(chunks, sparse_vecs)
}
with open(sparse_path, "w", encoding="utf-8") as f:
    json.dump(sparse_map, f)
print(f"Готово. Проиндексировано чанков: {len(chunks)}")