# CodeLens - умный поиск по кодовой базе на естественном языке

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

>**CodeLens** - RAG-система, способная отвечать на вопросы по кодовой базе на русском/английском, даже если в запросе не указаны конкретные имена функций или классов.


## Содержание

- [Возможности проекта](#возможности-проекта)
- [Быстрый старт](#быстрый-старт)
- [Запуск через Docker](#запуск-через-docker)
- [Запуск оценки системы](#запуск-оценки-системы)
- [Архитектура](#архитектура)
- [Технологический стек](#технологический-стек)
- [Стратегия чункования](#стратегия-чункования)
- [Метрики качества](#метрики-качества)
- [Структура проекта](#структура-проекта)



## Возможности проекта
- **Семантический поиск** по коду на естественном языке(RU/ENG)
- **Гибридный поиск**: комбинация плотных векторов и разреженных (BM25-подобных) весов с настраиваемым балансом `alpha`
- **Автоматический выбор `alpha`**: для русских запросов смещение в сторону векторного поиска (0.9), для английских - сбалансированный режим (0.5)
- **Подсветка синтаксиса** в результатах через `st.code(..., language="python")`
- **Оценка релевантности** каждого фрагмента в процентах
- **Дашборд метрик**: Precision@5 с визуализацией и прогрессом по каждому вопросу прямо в Streamlit
- **AI-анализ**: стриминговый ответ от Ollama (`mistral:7b`) на основе найденных фрагментов
- **Docker Compose** для запуска одной командой
## Быстрый старт

### №1 Установка зависимостей

```bash
pip install -r requirements.txt
```
> При первом запуске `index.py` или `app.py` будет установлена модель `BAAI/bge-m3` с Hugging Face.

### №2 Индексация кодовой базы

```bash
python index.py data/codebase_python
```
Скрипт обойдет все `.py` - файлы разобьет их на чанки и сохранит все в векторную базу данных (ChromaDB)

### №3 Запуск веб-интерфейса
```bash
streamlit run app.py
```
### №4 Запуск LLM-анализа

```bash
ollama serve
ollama pull mistral:7b
```

## Запуск через Docker
 
### Индексирование + запуск UI одной командой:
 
```bash

docker compose build
# Сначала индексируем (профиль index)
docker compose --profile index run --rm indexer
 
# Затем запускаем UI
docker compose up codelens

#остновка и удаление всех данных
docker compose down
docker compose down --volumes

#переиндексация
docker compose down
docker compose --profile index run --rm indexer
docker compose up
```
 
Приложение будет доступно на `http://localhost:8501`. Директории `chroma_db/` и `data/` монтируются как volumes, поэтому индекс сохраняется между перезапусками.

## Запуск оценки системы
### Запуск оценки через CLI
 
```bash
python eval.py
```
Скрипт выводит в консоль итоговый Precision@5, среднее время ответа и детальную таблицу по каждому вопросу.

### Запуск оценки через UI
 
Откройте вкладку **«Метрики Precision@5»** в Streamlit и нажмите «Запустить оценку». Результат отображается в виде метрик и таблицы с детализацией по каждому вопросу.

---
## Архитектура

```
┌────────────────────────────────────────────────────────────────┐
│                         index.py                               │
│  1. Обход директории → .py-файлы                               │
│  2. ast.parse → функции и классы (chunks)                      │
│  3. BAAI/bge-m3 → dense vector (1024d) + sparse lexical weights│
│  4. ChromaDB ← dense vectors + метаданные                      │
│  5. chroma_db/sparse_vectors.json ← sparse weights             │
└────────────────────────────────────────────────────────────────┘
              ↓ (один раз, при индексировании)
 
┌────────────────────────────────────────────────────────────────┐
│                         search.py                              │
│  Запрос → bge-m3 → dense + sparse вектор запроса               │
│  ChromaDB.query → топ-N кандидатов по dense                    │
│  Hybrid score = alpha * vec_norm + (1-alpha) * sparse_norm     │
│  Возврат топ-K чанков с релевантностью                         │
└────────────────────────────────────────────────────────────────┘
              ↓
 
┌────────────────────────────────────────────────────────────────┐
│                          app.py (Streamlit)                    │
│  Вкладка «Поиск»: поле ввода → результаты карточками           │
│  Сайдбар: top_k, alpha, включение AI-анализа (Ollama)          │
└────────────────────────────────────────────────────────────────┘
              ↓ (опционально)
 
┌────────────────────────────────────────────────────────────────┐
│                       answerLLM.py                             │
│  Найденные фрагменты + вопрос → ollama (mistral:7b) → стрим    │
└────────────────────────────────────────────────────────────────┘
```
 

## Технологический стек

| Слой | Инструмент | Примечание |
|---|---|---|
| Язык | Python 3.12 | |
| Парсинг | `ast` (stdlib) | Встроенный модуль, без доп. зависимостей |
| Эмбеддинги | `BAAI/bge-m3` via `FlagEmbedding` | Мультиязычная модель, dense + sparse в одном проходе |
| Векторная БД | ChromaDB 1.0.9 | Персистентный клиент, хранение на диске |
| Sparse-индекс | `sparse_vectors.json` | Сохраняется рядом с chroma_db |
| UI | Streamlit 1.45.1 | Две вкладки: поиск и метрики |
| LLM (опц.) | Ollama `mistral:7b` | Стриминг ответа, системный промпт на русском |
| Контейнеризация | Docker + Docker Compose | Профиль `index` для переиндексации |
 
---

## Стратегия чункования

**Один чанк = одна функция или класс**

Данное решение было принято так как: 

1. **Семантическая целостность** - функция является минимальной самостоятельной единицей в Python/JavaScript коде: у нее есть сигнатура, логически завершенное тело, docstring. Разбиение на более мелкие части нарушает контекст написанного кода, что может привести к неправильной трактовке кода.

2. **Совпадение модели поиска** - изучив кейс, мы пришли к выводу, что новички задают вопросы на подобии "Где реализованно X?", "Что делает функция/класс N?", "Зачем нужно T?". Ответы на данные вопросы помещаются в одну функцию. 

3. **Метаданные без потерь.** При чанковании по `ast.FunctionDef` / `ast.ClassDef` бесплатно получаем имя, номера строк и docstring - их можно сохранить в ChromaDB и показать пользователю без дополнительной обработки.

4. **Вложенность.** Методы классов индексируются отдельно (с квалифицированным именем `ClassName.method_name`) и дополнительно на уровне самого класса — это позволяет искать как конкретный метод, так и весь класс целиком.

**Ограничение длины эмбендингов** - было решение принятно решение ограничить длину эмбендингов, из-за возможного "размытия", для повышение результатов поиска.

## Метрики качества

1. как в проекте создаётся токен доступа и какой срок его жизни?

def create_access_token(subject: str | int, expires_minutes: int | None = None) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes or settings.access_token_expire_minutes
    )
    payload = {"sub": str(subject), "exp": expire}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)

2. how does the project verify a JWT token from an incoming request?

def get_token(token: str = Depends(oauth2_scheme)) -> TokenPayload:
    """
    Retrieve the token payload from the provided JWT token.

    Parameters:
        token (str, optional): The JWT token. Defaults to the value returned by the `oauth2_scheme` dependency.

    Returns:
        TokenPayload: The decoded token payload.

    Raises:
        HTTPException: If there is an error decoding the token or validating the payload.
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        token_data = TokenPayload(**payload)
    except (jwt.JWTError, ValidationError) as e:
        raise _get_credential_exception(status_code=status.HTTP_403_FORBIDDEN) from e
    return token_data

3. как в проекте реализована пагинация при получении списка объектов?

def get_pagination_params(skip: int = Query(0, ge=0), limit: int = Query(10, gt=0)) -> Tuple[int, int]:
    """
    Get the pagination parameters.

    Parameters:
        skip (int): The number of items to skip. Defaults to 0.
        limit (int): The maximum number of items to return. Defaults to 10.

    Returns:
        Tuple[int, int]: A tuple containing the skip and limit values.
    """
    return skip, limit

4. how does the project handle database session lifecycle for requests?

def get_ctx_db(database_url: str) -> Generator:
    """
    Context manager that creates a database session and yields
    it for use in a 'with' statement.

    Parameters:
        database_url (str): The URL of the database to connect to.

    Yields:
        Generator: A database session.

    Raises:
        Exception: If an error occurs while getting the database session.

    """
    log.debug("getting database session")
    db = get_local_session(database_url)()
    try:
        yield db
    except Exception as e:
        log.error("An error occurred while getting the database session. Error: %s", e)
        raise SQLAlchemyException from e
    finally:
        log.debug("closing database session")
        db.close()

5. где и как проект проверяет что пользователь является владельцем ресурса перед удалением?

def _is_admin(self, interaction: discord.Interaction) -> bool:
        """Проверка является ли пользователь администратором"""
        return interaction.user.guild_permissions.administrator or interaction.user.id == self.bot.owner_id

## Структура проекта

```
CODELENS/
├── index.py                  # Индексирующий скрипт
├── search.py                 # Гибридный поиск (dense + sparse)
├── app.py                    # Streamlit-приложение
├── answerLLM.py              # LLM-ответы через Ollama
├── eval.py                   # CLI-оценка Precision@5 и т.д
├── requirements.txt          # Зависимости
├── Dockerfile
├── docker-compose.yml
├── RostelecomLogo.png
├── .gitignore
├── data/
│   ├── codebase_python/
│   ├── gymhero/
│   ├── Discord-bot-Arvent/
│   ├── Discord-bot-Verant/
│   ├── encryptor_voice/
│   ├── task-semifinal-future-trajectory/
│   ├── tournament-system/
│   ├── UchiUnik/
│   ├── VTB_Multibanking/
│   └── eval_questions.json   # 15 вопросов с эталонными chunk_id
└── chroma_db/                # Персистентная векторная БД (генерируется)
    ├── chroma.sqlite3
    ├── sparse_vectors.json
    └── <collection_uuid>/
```