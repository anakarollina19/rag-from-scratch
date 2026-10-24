"""Interface web do RAG com Gradio.

Uso:
    python app.py        # abre em http://127.0.0.1:7860
"""
import os

import gradio as gr

from rag import config
from rag.embeddings import list_indexes
from rag.pipeline import RAG

rag = RAG()


def index_choices() -> list[tuple[str, str]]:
    """(nome exibido, caminho) para o menu de documentos."""
    return [(os.path.splitext(os.path.basename(p))[0], p) for p in list_indexes()]


def status_text() -> str:
    if not rag.index_path:
        return "Nenhum documento carregado. Envie um PDF para começar."
    name = os.path.splitext(os.path.basename(rag.index_path))[0]
    return (f"Documento ativo: **{name}** ({len(rag.chunks)} trechos indexados)  \n"
            f"Embeddings: `{rag.index_model}`")


def select_index(path: str):
    if path:
        rag.load(path)
    return status_text()


def upload_pdf(pdf_path: str, page_offset: float, model: str, progress=gr.Progress()):
    if not pdf_path:
        raise gr.Error("Escolha um arquivo PDF.")
    progress(0, desc="Lendo e indexando o PDF (pode levar alguns minutos)...")
    rag.embedding_model = model
    try:
        path = rag.index_pdf(pdf_path, page_offset=int(page_offset))
    except ValueError as e:
        raise gr.Error(str(e))
    return gr.update(choices=index_choices(), value=path), status_text()


def format_sources(chunks: list[dict]) -> str:
    lines = []
    for c in chunks:
        text = c["sentence_chunk"]
        if len(text) > 400:
            text = text[:400] + "…"
        rerank = f" · re-ranking {c['rerank_score']:.2f}" if "rerank_score" in c else ""
        lines.append(f"**Página {c['page_number']}** · score {c['score']:.3f}{rerank}\n\n> {text}")
    return "\n\n".join(lines)


def ask(question: str, k: float, generate: bool, rerank: bool):
    question = question.strip()
    if not question:
        raise gr.Error("Digite uma pergunta.")
    if not rag.index_path:
        raise gr.Error("Envie ou selecione um documento primeiro.")
    rag.rerank = rerank
    if not generate:
        return "*Geração desligada: veja abaixo os trechos encontrados pela busca.*", \
            format_sources(rag.search(question, int(k)))
    answer, context = rag.ask(question, int(k))
    return answer, format_sources(context)


with gr.Blocks(title="RAG from Scratch") as demo:
    gr.Markdown(
        "# 🔎 RAG from Scratch\n"
        "Envie um PDF, faça uma pergunta e veja a resposta do **Gemma 2B** "
        "junto com os trechos do documento usados para respondê-la. Tudo roda no seu computador."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Documento")
            index_menu = gr.Dropdown(label="Documentos já indexados", choices=index_choices())
            status = gr.Markdown()
            with gr.Accordion("Indexar um novo PDF", open=not list_indexes()):
                pdf_file = gr.File(label="PDF", file_types=[".pdf"], type="filepath")
                page_offset = gr.Number(
                    label="Páginas iniciais a descontar (capa, sumário...)", value=0, precision=0)
                model = gr.Dropdown(label="Modelo de embedding", choices=list(config.EMBEDDING_MODELS),
                                    value=config.EMBEDDING_MODEL,
                                    info="multilingual-e5 aceita perguntas em português")
                index_button = gr.Button("Indexar")

        with gr.Column(scale=2):
            gr.Markdown("### 2. Pergunta")
            question = gr.Textbox(label="Pergunta", lines=2,
                                  placeholder="How does saliva help with digestion?")
            with gr.Row():
                k = gr.Slider(1, 10, value=config.TOP_K, step=1, label="Trechos usados (k)")
                with gr.Column():
                    rerank = gr.Checkbox(value=config.RERANK, label="Re-ranking (cross-encoder)")
                    generate = gr.Checkbox(value=True, label="Gerar resposta com o Gemma (~1 min na CPU)")
            ask_button = gr.Button("Perguntar", variant="primary")
            answer_box = gr.Markdown(label="Resposta")
            with gr.Accordion("Fontes", open=True):
                sources_box = gr.Markdown()

    index_menu.change(select_index, index_menu, status)
    index_button.click(upload_pdf, [pdf_file, page_offset, model], [index_menu, status])
    ask_button.click(ask, [question, k, generate, rerank], [answer_box, sources_box])
    question.submit(ask, [question, k, generate, rerank], [answer_box, sources_box])

    def on_load():
        # Abre já com o documento indexado mais recente, se houver
        choices = index_choices()
        if choices and not rag.index_path:
            rag.load(choices[0][1])
        return gr.update(choices=choices, value=rag.index_path or None), status_text()

    demo.load(on_load, None, [index_menu, status])


if __name__ == "__main__":
    demo.launch()
