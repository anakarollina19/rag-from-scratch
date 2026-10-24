"""Interface web do RAG com Gradio.

Uso:
    python app.py        # abre em http://127.0.0.1:7860

Abas:
    Perguntar   chat com o documento ativo; cada resposta mostra as fontes
    Documentos  envio de PDF com pré-visualização e escolha das páginas a indexar
    Exportar    prévia, seleção e download (PDF/HTML) ou impressão das perguntas feitas
"""
import html
import os
from datetime import datetime

import gradio as gr

from rag import config
from rag.embeddings import list_indexes
from rag.pdf_preview import page_count, thumbnails
from rag.pipeline import RAG
from rag.report import build_html, html_to_pdf

EXPORT_DIR = "data/exports"
EXAMPLES = ["What is rickets and what causes it?", "O que causa o escorbuto?",
            "What is the glycemic index?", "O que é colostro?"]

rag = RAG()


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
    return (f"### 📘 {doc_name(rag.index_path)}\n"
            f"{len(rag.chunks)} trechos indexados · `{model}`")


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

def format_sources(chunks: list[dict]) -> str:
    lines = []
    for c in chunks:
        text = c["sentence_chunk"]
        text = text[:350] + "…" if len(text) > 350 else text
        lines.append(f"**Página {c['page_number']}** · similaridade {c['score']:.2f}\n> {text}")
    return "\n\n".join(lines)


def respond(question: str, chat: list, history: list, k: float, rerank: bool, generate: bool):
    """Gerador: vai atualizando o chat a cada etapa (busca, fontes, resposta)."""
    question = (question or "").strip()
    if not question:
        raise gr.Error("Digite uma pergunta.")
    if not rag.index_path:
        raise gr.Error("Envie ou selecione um documento primeiro.")

    chat = chat + [{"role": "user", "content": question},
                   {"role": "assistant", "content": "🔎 Buscando os trechos mais relevantes..."}]
    yield chat, history, ""

    rag.rerank = rerank
    sources = rag.search(question, int(k))
    chat[-1] = {"role": "assistant", "content": format_sources(sources),
                "metadata": {"title": f"📚 Fontes · {len(sources)} trechos", "status": "done"}}
    if generate:
        chat.append({"role": "assistant", "content": "✍️ Gerando a resposta com o Gemma (~1 min na CPU)..."})
        yield chat, history, ""
        from rag.generation import prompt_formatter
        answer = rag.generator.generate(prompt_formatter(question, sources))
        chat[-1] = {"role": "assistant", "content": answer}
    else:
        answer = "(Geração desligada: veja os trechos encontrados nas fontes.)"
        chat.append({"role": "assistant", "content": answer})

    history = history + [{"question": question, "answer": answer, "sources": sources,
                          "document": doc_name(rag.index_path), "time": f"{datetime.now():%H:%M}"}]
    yield chat, history, ""


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

THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])

CSS = """
.gradio-container { width: 100% !important; max-width: 1240px !important; margin: auto; }
#header { padding: 6px 4px 2px; }
#header h1 { font-size: 1.75rem; margin: 0; }
#header p { margin: 4px 0 0; opacity: .75; }
#doc-card { padding: 10px 14px 4px; }
#doc-card h3 { margin: 0 0 4px; word-break: break-word; }
#report-preview iframe { width: 100%; height: 620px; border: 1px solid var(--border-color-primary);
                         border-radius: 10px; background: #fff; }
.empty-preview { padding: 48px 16px; text-align: center; opacity: .7;
                 border: 1px dashed var(--border-color-primary); border-radius: 10px; }
"""

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

    gr.HTML("<div id='header'><h1>🔎 RAG from Scratch</h1>"
            "<p>Pergunte sobre um PDF e receba respostas do Gemma 2B com as páginas de origem. "
            "Tudo roda no seu computador.</p></div>")

    with gr.Tabs() as tabs:
        # ---- Perguntar ----
        with gr.Tab("💬 Perguntar", id="ask"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        height=540, show_label=False, buttons=["copy"],
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
                    with gr.Accordion("⚙️ Opções", open=True):
                        k = gr.Slider(1, 10, value=config.TOP_K, step=1, label="Trechos usados (k)")
                        rerank = gr.Checkbox(value=config.RERANK, label="Re-ranking",
                                             info="Mais preciso; +1,4 s por pergunta")
                        generate = gr.Checkbox(value=True, label="Gerar resposta com o Gemma",
                                               info="Desligue para ver só a busca (instantâneo)")
                    clear = gr.Button("🗑️ Limpar conversa", variant="secondary")

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
    chat_inputs = [question, chatbot, history, k, rerank, generate]
    chat_outputs = [chatbot, history, question]
    send.click(respond, chat_inputs, chat_outputs, show_progress="hidden")
    question.submit(respond, chat_inputs, chat_outputs, show_progress="hidden")
    clear.click(lambda: ([], []), None, [chatbot, history])

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
