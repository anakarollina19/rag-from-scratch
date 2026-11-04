"""Interface web do RAG com Gradio.

Uso:
    python app.py        # abre em http://127.0.0.1:7860

Abas:
    Perguntar   chat com o documento ativo, em dois modos:
                RAG (busca -> re-ranking -> resposta) ou Agente (o LLM escolhe as ferramentas)
    Análise     estatísticas, termos frequentes, mapa 2D dos embeddings (PCA) e temas (K-means)
    Documentos  envio de PDF com pré-visualização e escolha das páginas a indexar
    Exportar    prévia, seleção e download (PDF/HTML) ou impressão das perguntas feitas
"""
import html
import os
import random
import re
from datetime import datetime

import gradio as gr
import pandas as pd

from rag import config
from rag.agent import Agent
from rag.analysis import analyze
from rag.embeddings import list_indexes
from rag.pdf_preview import page_count, thumbnails
from rag.pipeline import RAG
from rag.report import build_html, html_to_pdf

EXPORT_DIR = "data/exports"
EXAMPLES = ["What is rickets and what causes it?", "O que causa o escorbuto?",
            "What is the glycemic index?", "O que é colostro?"]
MODES = ["⚡ RAG", "🤖 Agente"]

rag = RAG()
agent = Agent(rag)


# ---------- documentos ----------

def doc_name(index_path: str) -> str:
    return os.path.splitext(os.path.basename(index_path))[0]


def index_choices() -> list[tuple[str, str]]:
    """(nome exibido, caminho) para o menu de documentos."""
    return [(doc_name(p), p) for p in list_indexes()]


def doc_card() -> str:
    if not rag.index_path:
        return "### 📭 Nenhum documento\nEnvie um PDF na aba **Documentos**."
    model = rag.index_model.split("/")[-1]
    pages = len({c["page_number"] for c in rag.chunks})
    return (f"### 📘 {doc_name(rag.index_path)}\n"
            f"{len(rag.chunks)} trechos · {pages} páginas · `{model}`")


def select_doc(path: str):
    if path:
        rag.load(path)
    return doc_card()


def preview_pdf(pdf_path: str | None):
    """Ao escolher um arquivo: mostra as páginas e prepara o intervalo."""
    if not pdf_path:
        return "", gr.update(value=1, maximum=None), gr.update(value=1, maximum=None), [], \
            gr.update(interactive=False)
    n = page_count(pdf_path)
    size_mb = os.path.getsize(pdf_path) / 1e6
    info = f"**{os.path.basename(pdf_path)}** · {n} páginas · {size_mb:.1f} MB"
    return (info, gr.update(value=1, maximum=n), gr.update(value=n, maximum=n),
            thumbnails(pdf_path, 1, n), gr.update(interactive=True))


def preview_range(pdf_path: str | None, first: float, last: float):
    if not pdf_path or not first or not last:
        return [], ""
    first, last = int(first), int(last)
    if first > last:
        return [], "⚠️ A página inicial é maior que a final."
    n = last - first + 1
    note = f"Serão indexadas **{n} página(s)** (da {first} à {last})."
    if n > 24:
        note += " A prévia mostra as 24 primeiras."
    return thumbnails(pdf_path, first, last), note


def index_doc(pdf_path, first, last, model, page_offset, progress=gr.Progress()):
    if not pdf_path:
        raise gr.Error("Escolha um arquivo PDF.")
    first, last = int(first), int(last)
    if first > last:
        raise gr.Error("A página inicial é maior que a final.")
    progress(0.1, desc="Lendo e indexando o PDF...")
    total = page_count(pdf_path)
    rag.embedding_model = model
    try:
        path = rag.index_pdf(pdf_path, page_offset=int(page_offset or 0), first_page=first,
                             last_page=None if last == total else last)
    except ValueError as e:
        raise gr.Error(str(e))
    gr.Info(f"Documento pronto: {len(rag.chunks)} trechos indexados.")
    return (f"✅ **{doc_name(path)}** indexado ({len(rag.chunks)} trechos).",
            gr.update(choices=index_choices(), value=path), doc_card(), gr.Tabs(selected="ask"))


# ---------- perguntas ----------

def score_bar(value: float) -> str:
    """Barrinha de relevância (0 a 1), ex.: ▰▰▰▰▱ 0.82"""
    filled = max(0, min(5, round(value * 5)))
    return f"`{'▰' * filled}{'▱' * (5 - filled)}` similaridade {value:.2f}"


def format_sources(chunks: list[dict]) -> str:
    lines = []
    for c in chunks:
        text = c["sentence_chunk"]
        text = text[:350] + "…" if len(text) > 350 else text
        doc = f"{c['document']} · " if "document" in c else ""
        lines.append(f"**{doc}Página {c['page_number']}** · {score_bar(c['score'])}\n> {text}")
    return "\n\n".join(lines)


STAGES = [("🔎", "Busca vetorial"), ("⚖️", "Re-ranking"), ("🧩", "Prompt"), ("✍️", "Geração")]


def pipeline_html(active: int = -1, times: dict | None = None, skip: set | None = None) -> str:
    """Linha do tempo das etapas do RAG: concluídas, em andamento e puladas."""
    times, skip = times or {}, skip or set()
    items = []
    for i, (icon, name) in enumerate(STAGES):
        state = "skip" if i in skip else "done" if i < active else "active" if i == active else ""
        if active == len(STAGES) and i not in skip:
            state = "done"
        detail = times.get(i, "pulado" if i in skip else "")
        items.append(f"<div class='stage {state}'><div class='dot'>{icon}</div>"
                     f"<div><b>{name}</b><small>{detail}</small></div></div>")
    return f"<div class='pipeline'>{''.join(items)}</div>"


def respond(question: str, chat: list, history: list, mode: str, k: float, rerank: bool,
            generate: bool):
    """Gerador: vai atualizando o chat a cada etapa."""
    question = (question or "").strip()
    if not question:
        raise gr.Error("Digite uma pergunta.")
    if not rag.index_path:
        raise gr.Error("Envie ou selecione um documento primeiro.")
    rag.rerank = rerank
    chat = chat + [{"role": "user", "content": question}]
    if mode == MODES[1]:
        yield from respond_agent(question, chat, history, int(k))
        return

    import time
    times, skip = {}, set() if rerank else {1}
    if not generate:
        skip |= {2, 3}
    chat.append({"role": "assistant", "content": "🔎 Buscando os trechos mais relevantes..."})
    yield chat, history, "", pipeline_html(0, times, skip)

    start = time.perf_counter()
    sources = rag.search(question, int(k))
    elapsed = time.perf_counter() - start
    times[0] = f"{len(sources)} trechos"
    if rerank:
        times[1] = f"{elapsed:.1f} s"
    chat[-1] = {"role": "assistant", "content": format_sources(sources),
                "metadata": {"title": f"📚 Fontes · {len(sources)} trechos", "status": "done"}}
    if generate:
        chat.append({"role": "assistant", "content": "✍️ Gerando a resposta (~1 min na CPU)..."})
        times[2] = "pronto"
        yield chat, history, "", pipeline_html(3, times, skip)
        from rag.generation import prompt_formatter
        start = time.perf_counter()
        answer = rag.generator.generate(prompt_formatter(question, sources))
        times[3] = f"{time.perf_counter() - start:.0f} s"
        chat[-1] = {"role": "assistant", "content": answer}
    else:
        answer = "(Geração desligada: veja os trechos encontrados nas fontes.)"
        chat.append({"role": "assistant", "content": answer})

    history = history + [{"question": question, "answer": answer, "sources": sources,
                          "document": doc_name(rag.index_path), "time": f"{datetime.now():%H:%M}"}]
    yield chat, history, "", pipeline_html(len(STAGES), times, skip)


AGENT_TITLES = {"thought": "🧠 Pensamento", "action": "🔧 Ação", "observation": "👁️ Observação"}


def respond_agent(question: str, chat: list, history: list, k: int):
    """Mostra cada passo do agente (pensamento, ação, observação) conforme acontece."""
    agent.k = k
    sources, step_n = [], 0
    chat.append({"role": "assistant", "content": "🤖 O agente está planejando o próximo passo..."})
    yield chat, history, "", agent_html(step_n)
    answer = ""
    for step in agent.run(question):
        if step.kind == "final":
            answer = step.content
            chat[-1] = {"role": "assistant", "content": answer}
            break
        if step.kind == "action":
            step_n += 1
            title = f"🔧 Passo {step_n} · {step.tool}"
            content = f"`{step.tool}[{step.content}]`"
        elif step.kind == "observation":
            sources += step.sources or []
            title = f"👁️ Resultado de {step.tool}"
            content = format_sources(step.sources) if step.sources else step.content
        else:
            title, content = AGENT_TITLES["thought"], step.content or "…"
        chat[-1] = {"role": "assistant", "content": content,
                    "metadata": {"title": title, "status": "done"}}
        chat.append({"role": "assistant", "content": "🤖 Pensando..."})
        yield chat, history, "", agent_html(step_n)

    history = history + [{"question": question, "answer": answer, "sources": sources,
                          "document": doc_name(rag.index_path), "time": f"{datetime.now():%H:%M}"}]
    yield chat, history, "", agent_html(step_n, done=True)


def agent_html(steps: int, done: bool = False) -> str:
    status = "✅ resposta pronta" if done else "⏳ trabalhando"
    tools = " ".join(f"<code>{t}</code>" for t in agent.tools)
    return (f"<div class='pipeline agent'><div class='stage {'done' if done else 'active'}'>"
            f"<div class='dot'>🤖</div><div><b>Agente ReAct · {steps} ferramenta(s) usada(s)</b>"
            f"<small>{status} · ferramentas: {tools}</small></div></div></div>")


def suggest_questions():
    """Pede ao LLM 3 perguntas sobre trechos sorteados do documento."""
    if not rag.chunks:
        raise gr.Error("Envie ou selecione um documento primeiro.")
    sample = random.sample(rag.chunks, min(3, len(rag.chunks)))
    context = "\n".join(f"- {c['sentence_chunk'][:600]}" for c in sample)
    prompt = (f"Context:\n{context}\n\nWrite 3 short, interesting questions a student could ask "
              "that this context answers. One question per line, no numbering.")
    text = rag.generator.generate(prompt, max_new_tokens=120)
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", l).strip() for l in text.splitlines()]
    questions = [l for l in lines if l.endswith("?")][:3]
    questions += [""] * (3 - len(questions))
    return [gr.update(value=q, visible=bool(q)) for q in questions]


# ---------- análise ----------

def run_analysis(auto_k: bool, k: float, history: list):
    if rag.embeddings is None:
        raise gr.Error("Envie ou selecione um documento primeiro.")
    result = analyze(rag.chunks, rag.embeddings, None if auto_k else int(k))

    cards = "".join(f"<div class='stat'><small>{name}</small><b>{value}</b></div>"
                    for name, value in result["stats"].items())
    terms = pd.DataFrame(result["terms"], columns=["termo", "frequência"])

    # fontes da última pergunta aparecem destacadas no mapa
    last = {(s["page_number"], s["sentence_chunk"]) for s in (history[-1]["sources"] if history else [])}
    names = result["names"]
    points = pd.DataFrame({
        "PC1": result["points"][:, 0].tolist(), "PC2": result["points"][:, 1].tolist(),
        "tema": [("★ fontes da última pergunta" if (c["page_number"], c["sentence_chunk"]) in last
                  else f"{l + 1}. {names[l]}") for c, l in zip(rag.chunks, result["labels"].tolist())],
        "página": [c["page_number"] for c in rag.chunks],
        "trecho": [c["sentence_chunk"][:120] + "…" for c in rag.chunks],
    })

    sizes = pd.Series(result["labels"].tolist()).value_counts().sort_index()
    topics = pd.DataFrame({"grupo": [i + 1 for i in sizes.index],
                           "termos característicos (TF-IDF)": [names[i] for i in sizes.index],
                           "trechos": sizes.values})
    sil = pd.DataFrame([(k_, s) for k_, s in result["silhouette"].items() if s is not None],
                       columns=["k", "silhueta"])
    var = result["variance"]
    note = (f"**{result['k']} temas** encontrados com K-means"
            f"{' (k escolhido pela maior silhueta)' if auto_k else ''}. "
            f"O mapa usa PCA: os 2 eixos explicam {100 * (var[0] + var[1]):.1f}% da variância "
            f"dos embeddings de {rag.embeddings.shape[1]} dimensões. "
            "Trechos próximos no mapa têm significado parecido.")
    return (f"<div class='stats'>{cards}</div>", terms, points, topics,
            gr.update(value=sil, visible=not sil.empty), note)


# ---------- exportação ----------

def export_choices(history: list):
    choices = [(f"{i + 1}. {e['question']}", i) for i, e in enumerate(history)]
    return gr.update(choices=choices, value=list(range(len(history))))


def selected_entries(history: list, selected: list) -> list[dict]:
    return [history[i] for i in sorted(selected or []) if i < len(history)]


def export_preview(history: list, selected: list, include_sources: bool) -> str:
    entries = selected_entries(history, selected)
    if not entries:
        return ("<div class='empty-preview'>Nenhuma pergunta selecionada.<br>"
                "Faça perguntas na aba <b>Perguntar</b> e elas aparecem aqui.</div>")
    page = build_html(entries, include_sources)
    return f"<iframe srcdoc=\"{html.escape(page, quote=True)}\"></iframe>"


def export_file(history: list, selected: list, include_sources: bool, fmt: str):
    entries = selected_entries(history, selected)
    if not entries:
        raise gr.Error("Selecione pelo menos uma pergunta.")
    os.makedirs(EXPORT_DIR, exist_ok=True)
    path = f"{EXPORT_DIR}/perguntas-{datetime.now():%Y%m%d-%H%M%S}.{fmt}"
    page = build_html(entries, include_sources)
    if fmt == "pdf":
        html_to_pdf(page, path)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(page)
    return gr.update(value=path, visible=True)


# ---------- layout ----------

THEME = gr.themes.Soft(primary_hue="violet", secondary_hue="cyan", neutral_hue="slate",
                       radius_size="lg",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
                       font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"])

CSS = """
.gradio-container { width: 100% !important; max-width: 1280px !important; margin: auto; }

/* cabeçalho */
#hero { position: relative; overflow: hidden; padding: 28px 28px 22px; border-radius: 20px;
        color: #fff; margin-bottom: 6px;
        background: radial-gradient(circle at 85% 20%, #22d3ee55, transparent 45%),
                    linear-gradient(120deg, #4c1d95, #6d28d9 45%, #0e7490); }
#hero h1 { font-size: 2rem; margin: 0; letter-spacing: -.02em; color: #fff; }
#hero p { margin: 6px 0 14px; opacity: .85; max-width: 760px; color: #fff; }
#hero .pills span { display: inline-block; margin: 0 6px 6px 0; padding: 4px 11px;
                    border-radius: 999px; font-size: .78rem; background: #ffffff22;
                    border: 1px solid #ffffff33; backdrop-filter: blur(4px); }
#theme-btn { position: absolute; top: 14px; right: 14px; }

/* linha do tempo das etapas */
.pipeline { display: flex; gap: 8px; flex-wrap: wrap; }
.stage { flex: 1 1 140px; display: flex; gap: 10px; align-items: center; padding: 8px 12px;
         border-radius: 12px; border: 1px solid var(--border-color-primary);
         background: var(--block-background-fill); opacity: .55; transition: all .3s; }
.stage .dot { font-size: 1.2rem; }
.stage b { display: block; font-size: .85rem; }
.stage small { opacity: .7; font-size: .75rem; }
.stage.done { opacity: 1; border-color: #10b98188; }
.stage.active { opacity: 1; border-color: var(--color-accent);
                box-shadow: 0 0 0 3px var(--color-accent-soft); animation: pulse 1.4s infinite; }
.stage.skip { opacity: .35; text-decoration: line-through; }
@keyframes pulse { 50% { box-shadow: 0 0 0 6px transparent; } }

/* cartões de estatística */
.stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
.stat { padding: 14px; border-radius: 14px; border: 1px solid var(--border-color-primary);
        background: var(--block-background-fill); }
.stat small { display: block; opacity: .65; font-size: .75rem; }
.stat b { font-size: 1.35rem; }

#doc-card { padding: 10px 14px 4px; }
#doc-card h3 { margin: 0 0 4px; word-break: break-word; }
.suggestion { text-align: left !important; justify-content: flex-start !important; }
#report-preview iframe { width: 100%; height: 620px; border: 1px solid var(--border-color-primary);
                         border-radius: 10px; background: #fff; }
.empty-preview { padding: 48px 16px; text-align: center; opacity: .7;
                 border: 1px dashed var(--border-color-primary); border-radius: 10px; }
@media (max-width: 640px) { #hero { padding: 20px 16px; } #hero h1 { font-size: 1.5rem; } }
"""

TOGGLE_THEME_JS = "() => { document.body.classList.toggle('dark'); }"

# Impede o Chrome de traduzir a página automaticamente (botões viravam "r")
HEAD = """
<meta name="google" content="notranslate">
<script>
  document.documentElement.setAttribute("translate", "no");
  document.documentElement.lang = "pt-BR";
</script>
"""

PRINT_JS = """
() => {
  const frame = document.querySelector('#report-preview iframe');
  if (frame) { frame.contentWindow.focus(); frame.contentWindow.print(); }
  else { alert('Selecione pelo menos uma pergunta para imprimir.'); }
}
"""

with gr.Blocks(title="RAG from Scratch") as demo:
    history = gr.State([])  # perguntas feitas: [{question, answer, sources, document, time}]

    with gr.Column(elem_id="hero"):
        gr.HTML("<h1>🔎 RAG from Scratch</h1>"
                "<p>Converse com qualquer PDF: busca semântica, re-ranking, um agente que escolhe "
                "suas próprias ferramentas e uma análise exploratória dos dados do documento. "
                "Tudo roda no seu computador.</p>"
                "<div class='pills'><span>🧠 LLM local</span><span>📚 RAG do zero</span>"
                "<span>🤖 Agente ReAct</span><span>📊 PCA + K-means</span>"
                "<span>📏 Avaliação hit@3 / MRR</span><span>🌎 PT · EN</span></div>")
        theme_btn = gr.Button("🌓", elem_id="theme-btn", size="sm", min_width=40, scale=0)

    with gr.Tabs() as tabs:
        # ---- Perguntar ----
        with gr.Tab("💬 Perguntar", id="ask"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    mode = gr.Radio(MODES, value=MODES[0], show_label=False,
                                    info="RAG: busca e responde direto · Agente: o LLM decide "
                                         "quais ferramentas usar (buscar, ler página, comparar "
                                         "documentos, calcular)")
                    stages = gr.HTML(pipeline_html())
                    chatbot = gr.Chatbot(
                        height=500, show_label=False, buttons=["copy"],
                        placeholder="### Pergunte algo sobre o documento\n"
                                    "Funciona em português e inglês. Cada resposta mostra as fontes.")
                    with gr.Row():
                        question = gr.Textbox(placeholder="Digite sua pergunta e aperte Enter...",
                                              show_label=False, scale=8, autofocus=True)
                        send = gr.Button("Enviar", variant="primary", scale=1, min_width=90)
                    gr.Examples(EXAMPLES, inputs=question, label="Exemplos (livro de nutrição)")

                with gr.Column(scale=1, min_width=270):
                    with gr.Group():
                        doc_status = gr.Markdown(elem_id="doc-card")
                        doc_menu = gr.Dropdown(label="Trocar documento", choices=index_choices())
                    with gr.Accordion("✨ Perguntas sugeridas pela IA", open=True):
                        suggest_btn = gr.Button("Sugerir perguntas sobre o documento", size="sm")
                        suggestions = [gr.Button(visible=False, size="sm", variant="secondary",
                                                 elem_classes="suggestion") for _ in range(3)]
                    with gr.Accordion("⚙️ Opções", open=False):
                        k = gr.Slider(1, 10, value=config.TOP_K, step=1, label="Trechos usados (k)")
                        rerank = gr.Checkbox(value=config.RERANK, label="Re-ranking",
                                             info="Mais preciso; +1,4 s por pergunta")
                        generate = gr.Checkbox(value=True, label="Gerar resposta com o Gemma",
                                               info="Desligue para ver só a busca (instantâneo)")
                    clear = gr.Button("🗑️ Limpar conversa", variant="secondary")

        # ---- Análise ----
        with gr.Tab("📊 Análise", id="analysis"):
            gr.Markdown("Análise exploratória do documento ativo: estatísticas, termos mais "
                        "frequentes, mapa dos embeddings (PCA) e temas encontrados por K-means. "
                        "Tudo implementado em PyTorch, sem scikit-learn.")
            with gr.Row():
                auto_k = gr.Checkbox(value=True, label="Escolher o número de temas automaticamente",
                                     info="Testa k de 2 a 8 e fica com a maior silhueta")
                k_topics = gr.Slider(2, 12, value=5, step=1, label="Número de temas (k)")
                analyze_btn = gr.Button("📊 Analisar documento", variant="primary", scale=0,
                                        min_width=200)
            stats_html = gr.HTML()
            analysis_note = gr.Markdown()
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    scatter = gr.ScatterPlot(x="PC1", y="PC2", color="tema",
                                             tooltip=["página", "tema", "trecho"],
                                             title="Mapa semântico dos trechos (PCA 2D)",
                                             height=460)
                with gr.Column(scale=2):
                    terms_plot = gr.BarPlot(x="termo", y="frequência", sort="-y",
                                            title="Termos mais frequentes", height=460)
            with gr.Row(equal_height=False):
                topics_table = gr.Dataframe(label="Temas (K-means)", interactive=False, scale=3)
                sil_plot = gr.LinePlot(x="k", y="silhueta", title="Silhueta por k (maior = melhor)",
                                       height=260, visible=False, scale=2)

        # ---- Documentos ----
        with gr.Tab("📄 Documentos", id="docs"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    pdf_file = gr.File(label="Enviar PDF", file_types=[".pdf"], type="filepath")
                    pdf_info = gr.Markdown()
                    with gr.Row():
                        first_page = gr.Number(label="Da página", value=1, precision=0, minimum=1)
                        last_page = gr.Number(label="Até a página", value=1, precision=0, minimum=1)
                    range_note = gr.Markdown()
                    with gr.Accordion("Avançado", open=False):
                        model = gr.Dropdown(label="Modelo de embedding",
                                            choices=list(config.EMBEDDING_MODELS),
                                            value=config.EMBEDDING_MODEL,
                                            info="O multilingual-e5 aceita perguntas em português")
                        page_offset = gr.Number(
                            label="Descontar na numeração", value=0, precision=0,
                            info="Ex.: 12 faz a página 13 do PDF aparecer como página 1")
                    index_btn = gr.Button("📥 Indexar documento", variant="primary", interactive=False)
                    index_status = gr.Markdown()
                with gr.Column(scale=3):
                    gallery = gr.Gallery(label="Pré-visualização", columns=4, height=600,
                                         object_fit="contain", allow_preview=True)

        # ---- Exportar ----
        with gr.Tab("🖨️ Exportar", id="export"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=300):
                    selected = gr.CheckboxGroup(label="Perguntas incluídas", choices=[])
                    include_sources = gr.Checkbox(value=True, label="Incluir as fontes")
                    with gr.Row():
                        pdf_btn = gr.Button("⬇️ PDF", variant="primary")
                        html_btn = gr.Button("⬇️ HTML")
                    print_btn = gr.Button("🖨️ Imprimir")
                    export_out = gr.File(label="Arquivo gerado", visible=False)
                with gr.Column(scale=2):
                    preview = gr.HTML(elem_id="report-preview")

    # ---- eventos ----
    chat_inputs = [question, chatbot, history, mode, k, rerank, generate]
    chat_outputs = [chatbot, history, question, stages]
    send.click(respond, chat_inputs, chat_outputs, show_progress="hidden")
    question.submit(respond, chat_inputs, chat_outputs, show_progress="hidden")
    clear.click(lambda: ([], [], pipeline_html()), None, [chatbot, history, stages])
    mode.change(lambda m: agent_html(0, done=True) if m == MODES[1] else pipeline_html(),
                mode, stages)
    theme_btn.click(None, js=TOGGLE_THEME_JS)

    suggest_btn.click(suggest_questions, None, suggestions)
    for b in suggestions:
        b.click(lambda q: q, b, question)

    analyze_btn.click(run_analysis, [auto_k, k_topics, history],
                      [stats_html, terms_plot, scatter, topics_table, sil_plot, analysis_note])

    doc_menu.change(select_doc, doc_menu, doc_status)
    pdf_file.change(preview_pdf, pdf_file, [pdf_info, first_page, last_page, gallery, index_btn])
    for box in (first_page, last_page):
        box.blur(preview_range, [pdf_file, first_page, last_page], [gallery, range_note])
        box.submit(preview_range, [pdf_file, first_page, last_page], [gallery, range_note])
    index_btn.click(index_doc, [pdf_file, first_page, last_page, model, page_offset],
                    [index_status, doc_menu, doc_status, tabs], show_progress_on=[index_status])

    history.change(export_choices, history, selected)
    for trigger in (selected.change, include_sources.change):
        trigger(export_preview, [history, selected, include_sources], preview)
    pdf_btn.click(lambda h, s, i: export_file(h, s, i, "pdf"), [history, selected, include_sources], export_out)
    html_btn.click(lambda h, s, i: export_file(h, s, i, "html"), [history, selected, include_sources], export_out)
    print_btn.click(None, js=PRINT_JS)

    def on_load():
        # Abre já com o documento indexado mais recente, se houver
        choices = index_choices()
        if choices and not rag.index_path:
            rag.load(choices[0][1])
        return (gr.update(choices=choices, value=rag.index_path or None), doc_card(),
                export_preview([], [], True))

    demo.load(on_load, None, [doc_menu, doc_status, preview])


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, head=HEAD)
