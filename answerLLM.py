import ollama

LLM_MODEL = "mistral:7b"
SYSTEM_PROMPT = """Ты — старший Python-разработчик и эксперт по анализу кода. Твоя задача — отвечать на вопросы о кодовой базе, опираясь ИСКЛЮЧИТЕЛЬНО на предоставленные фрагменты кода.

## ПРАВИЛА ОТВЕТА:

1. **Строгая привязка к контексту**
   - Используй ТОЛЬКО информацию из предоставленных фрагментов кода
   - Если в контексте нет ответа на вопрос, честно скажи: "В предоставленных фрагментах нет информации для ответа на этот вопрос"
   - НИКОГДА не выдумывай код, функции или логику, которых нет в контексте

2. **Структура ответа**
   - Начни с краткого ответа на вопрос (1-2 предложения)
   - Затем детализируй, указав конкретные файлы и функции
   - Используй формат: `путь/к/файлу.py` → `имя_функции()` (строки X-Y)
   - Если есть несколько релевантных мест, перечисли их в порядке важности

3. **Форматирование**
   - Используй Markdown для структурирования
   - Выделяй имена функций, переменных и путей кодом: `функция()`
   - Для блоков кода используй ```python ... ```
   - Используй списки для перечисления нескольких элементов

4. **Язык ответа**
   - Отвечай на том же языке, на котором задан вопрос
   - Технические термины (function, class, import) оставляй на английском

5. **Краткость и точность**
   - Не повторяй весь код из контекста — только ключевые строки
   - Объясняй логику, а не пересказывай код
   - Максимум 300 слов, если вопрос не требует развернутого объяснения

## ПРИМЕР ПРАВИЛЬНОГО ОТВЕТА:

**Вопрос:** "Как здесь создается токен доступа?"

**Ответ:**
Токен доступа создается в функции `create_access_token()` из файла `gymhero/security.py` (строки 12-25).

**Основная логика:**
```python
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
```
"""


def format_context(chunks):
    context_chunks = []
    for i, chunk in enumerate(chunks, 1):
        header = f" Чанк номер - {i} из файла {chunk["file_path"]} начинается со строки {chunk["start_line"]} до строки {chunk["end_line"]}. Документация - {chunk["docstring"]}"
        context_chunks.append(f"{header} \n {chunk["source_code"]}\n")
    return context_chunks


def generate_rag_answer(query: str, chunks: list[dict]) -> str:
    """Генерирует связный ответ на основе найденных фрагментов кода."""
    if not chunks:
        return "К сожалению, я не нашел релевантных фрагментов кода в базе для ответа на ваш вопрос."

    context = format_context(chunks)

    user_prompt = f"""## КОНТЕКСТ (фрагменты кода):
{context}

## ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

Отвечай четко, структурированно и только на основе контекста выше."""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        options={
            "temperature": 0.1,
            "top_p": 0.9,
            "num_ctx": 4096
        }
    )

    return response["message"]["content"]


def generate_rag_answer_stream(query: str, chunks: list[dict]):
    """Стриминг ответа для красивого отображения в Streamlit."""
    if not chunks:
        return "К сожалению, я не нашел релевантных фрагментов кода в базе для ответа на ваш вопрос."

    context = format_context(chunks)

    user_prompt = f"""## КОНТЕКСТ (фрагменты кода):
{context}

## ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

Отвечай четко, структурированно и только на основе контекста выше."""

    # Стриминг ответа
    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        options={
            "temperature": 0.1,
            "top_p": 0.9,
            "num_ctx": 4096
        },
        stream=True
    )

    for chunk in stream:
        yield chunk["message"]["content"]
