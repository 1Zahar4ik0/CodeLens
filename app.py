import json
import streamlit as st

from answerLLM import generate_rag_answer_stream

st.set_page_config(
    page_title="CodeLens",
    # page_icon="RostelecomLogo.png",
    page_icon="",
    layout="wide"
)

from search import search

def _chunks_match(predicted: str, reference: str, tolerance: int = 2) -> bool:
    p_parts = predicted.rsplit(":", 2)
    r_parts = reference.rsplit(":", 2)
    if len(p_parts) != 3 or len(r_parts) != 3:
        return False
    try:
        return (p_parts[0] == r_parts[0]
                and p_parts[1] == r_parts[1]
                and abs(int(p_parts[2]) - int(r_parts[2])) <= tolerance)
    except ValueError:
        return False

def _count_matches(found_ids: list[str], correct: list[str]) -> int:
    used = set()
    hits = 0
    for pred in found_ids:
        for j, ref in enumerate(correct):
            if j not in used and _chunks_match(pred, ref):
                hits += 1
                used.add(j)
                break
    return hits

st.sidebar.title("Настройки")

top_k = st.sidebar.slider(
    label="Количество результатов",
    min_value=1,
    max_value=10,
    value=5,
)

alpha = st.sidebar.slider(
    label="Баланс поиска (вектор <=> ключевые слова)",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1,
    help="1.0 — только векторный поиск. 0.0 — только по ключевым словам.",
)

enable_llm = st.sidebar.checkbox(
    label="Включить AI-анализ",
    value=False,
    help="Генерировать связный ответ на основе найденных фрагментов кода"
)

tab_search, tab_metrics = st.tabs(["Поиск", "Метрики Precision@5"])

with tab_search:
    st.title("CodeLens")
    st.caption("Умный поиск по кодовой базе")
    st.divider()

    query = st.text_input(
        label="Введите запрос:",
        placeholder="как обрабатываются ошибки авторизации?",
    )

    search_button = st.button("Найти", type="primary")

    if search_button and not query:
        st.warning("Строка запроса не должна быть пустой")

    if search_button and query:
        with st.spinner("Ищем похожие фрагменты..."):
            results = search(query, top_k=top_k, alpha=alpha)

        st.markdown(f"Найдено результатов: **{len(results)}**")
        st.divider()

        for r in results:
            col_info, col_relevance = st.columns([4, 1])

            with col_info:
                st.markdown(f"**{r['type'].capitalize()}** `{r['name']}`")
                st.caption(
                    f"{r['file_path']} · строки {r['start_line']}–{r['end_line']}")

            with col_relevance:
                st.metric(label="Релевантность", value=f"{r['relevance']}%")

            if r["docstring"]:
                st.info(r["docstring"])

            st.code(r["source_code"], language="python")
            st.divider()

        if enable_llm and results:
            st.markdown("---")
            st.subheader("Анализ от ИИ")
            st.caption("Нейросеть анализирует найденные фрагменты и формирует связаный ответ...")
            
            answer_placeholder = st.empty()
            full_answer = ""

            for token in generate_rag_answer_stream(query, results):
                full_answer += token
                answer_placeholder.markdown(full_answer + "▌")

            answer_placeholder.markdown(full_answer)

with tab_metrics:
    st.title("Оценка качества поиска")
    st.caption("Precision@5 — доля правильных ответов")
    st.divider()

    eval_path = "data/eval_questions.json"
    run_button = st.button("Запустить оценку", type="primary")

    if run_button:
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                questions = json.load(f)
        except FileNotFoundError:
            st.error(f"Файл не найден: {eval_path}")
            st.stop()

        total_precision = 0.0
        rows = []
        progress = st.progress(0, text="Оцениваем...")

        for i, item in enumerate(questions):
            question = item["query"]
            correct = item["correct_chunk_ids"]

            results = search(question, top_k=5, alpha=alpha)
            found_ids = [r["chunk_id"] for r in results]

            hits = _count_matches(found_ids, correct)
            n = min(5, len(correct))
            precision = round(hits / n * 100, 1)
            total_precision += precision

            rows.append({
                "Вопрос":      question,
                "Попаданий":   f"{hits}/{len(correct)}",
                "Precision@5": f"{precision}%",
            })

            progress.progress(
                (i + 1) / len(questions),
                text=f"Вопрос {i + 1} из {len(questions)}",
            )

        progress.empty()

        overall = round(total_precision / len(questions), 1)
        target = 60.0

        col_metric, col_target = st.columns(2)
        with col_metric:
            st.metric(
                label="Итоговый Precision@5",
                value=f"{overall}%",
                delta=f"{round(overall - target, 1)}% до цели",
            )
        with col_target:
            st.metric(label="Целевое значение", value="60%")

        if overall >= target:
            st.success(f"Победа, {overall}% ≥ 60%")
        else:
            st.error(f"Поражение, {overall}% < 60%")

        st.divider()
        st.subheader("Детали по каждому вопросу")
        st.table(rows)