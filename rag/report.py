"""Relatório das perguntas feitas: HTML (para visualizar e imprimir) e PDF."""
import html
from datetime import datetime

import pymupdf

STYLE = """
body { font-family: sans-serif; font-size: 11pt; color: #1f2328; line-height: 1.45; }
h1 { font-size: 18pt; margin: 0 0 4pt 0; }
.meta { color: #59636e; font-size: 9pt; margin-bottom: 14pt; }
.qa { margin-bottom: 16pt; padding-bottom: 10pt; border-bottom: 1px solid #d1d9e0; }
.q { font-weight: bold; font-size: 12pt; margin-bottom: 4pt; }
.a { white-space: pre-wrap; }
.src { color: #59636e; font-size: 9pt; margin-top: 6pt; }
.src p { margin: 2pt 0; }
"""


def build_html(entries: list[dict], include_sources: bool = True,
               title: str = "Perguntas e respostas") -> str:
    """entries: [{"question", "answer", "sources": [{"page_number", "sentence_chunk"}], "document", "time"}]"""
    docs = sorted({e["document"] for e in entries})
    parts = [f"<h1>{html.escape(title)}</h1>",
             f"<div class='meta'>Gerado em {datetime.now():%d/%m/%Y %H:%M} · "
             f"{len(entries)} pergunta(s) · Documento(s): {html.escape(', '.join(docs)) or '-'}</div>"]
    for i, e in enumerate(entries, 1):
        parts.append("<div class='qa'>")
        parts.append(f"<div class='q'>{i}. {html.escape(e['question'])}</div>")
        parts.append(f"<div class='a'>{html.escape(e['answer'])}</div>")
        if include_sources and e.get("sources"):
            items = "".join(
                f"<p><b>p. {s['page_number']}</b>: {html.escape(s['sentence_chunk'][:220])}…</p>"
                for s in e["sources"])
            parts.append(f"<div class='src'>Fontes ({html.escape(e['document'])}):{items}</div>")
        parts.append("</div>")
    body = "\n".join(parts)
    return (f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title><style>{STYLE}</style></head>"
            f"<body>{body}</body></html>")


def html_to_pdf(page_html: str, path: str) -> str:
    """Converte o HTML do relatório em PDF A4 usando o próprio PyMuPDF."""
    story = pymupdf.Story(html=page_html)
    page_rect = pymupdf.paper_rect("a4")
    content = page_rect + (50, 50, -50, -50)  # margens de 50 pt
    writer = pymupdf.DocumentWriter(path)
    more = True
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(content)
        story.draw(device)
        writer.end_page()
    writer.close()
    return path
