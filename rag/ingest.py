"""Etapa 1: baixar o PDF, extrair texto por página e quebrar em chunks de sentenças."""
import os
import re

import pymupdf
import requests
from spacy.lang.en import English
from tqdm.auto import tqdm

from . import config


def download_pdf(path: str = config.BOOK_PATH, url: str = config.BOOK_URL) -> str:
    """Baixa o PDF caso ainda não exista localmente."""
    if os.path.exists(path):
        print(f"[INFO] {path} já existe.")
        return path
    print(f"[INFO] Baixando {url} ...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(response.content)
    print(f"[INFO] Salvo em {path}")
    return path


def text_formatter(text: str) -> str:
    """Limpeza mínima do texto."""
    return text.replace("\n", " ").strip()


def open_and_read_pdf(pdf_path: str, page_offset: int = 0,
                      first_page: int = 1, last_page: int | None = None) -> list[dict]:
    """Lê o PDF página por página e coleta estatísticas básicas.

    page_offset desconta páginas iniciais (capa, sumário...) para que a
    numeração bata com a impressa no documento. Com 0, vale a do PDF (1, 2, 3...).
    first_page/last_page (contando do 1, inclusive) limitam o trecho do PDF lido.
    """
    doc = pymupdf.open(pdf_path)
    last_page = min(last_page or len(doc), len(doc))
    pages = []
    for index in tqdm(range(first_page - 1, last_page), desc="Lendo PDF"):
        text = text_formatter(doc[index].get_text())
        pages.append({
            "page_number": index + 1 - page_offset,
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_token_count": len(text) / 4,  # ~4 caracteres por token
            "text": text,
        })
    return pages


def split_list(items: list[str], size: int) -> list[list[str]]:
    """Ex.: 17 sentenças com size=10 -> [[10], [7]]."""
    return [items[i:i + size] for i in range(0, len(items), size)]


def chunk_pages(pages: list[dict],
                chunk_size: int = config.SENTENCES_PER_CHUNK,
                min_tokens: int = config.MIN_TOKEN_LENGTH) -> list[dict]:
    """Divide cada página em sentenças (spaCy) e agrupa em chunks."""
    nlp = English()
    nlp.add_pipe("sentencizer")

    chunks = []
    for page in tqdm(pages, desc="Gerando chunks"):
        sentences = [str(s) for s in nlp(page["text"]).sents]
        for group in split_list(sentences, chunk_size):
            text = " ".join(group).replace("  ", " ").strip()
            text = re.sub(r"\.([A-Z])", r". \1", text)  # ".A" -> ". A"
            token_count = len(text) / 4
            if token_count <= min_tokens:
                continue  # descarta chunks muito curtos (cabeçalhos, rodapés...)
            chunks.append({
                "page_number": page["page_number"],
                "sentence_chunk": text,
                "chunk_char_count": len(text),
                "chunk_word_count": len(text.split(" ")),
                "chunk_token_count": token_count,
            })
    return chunks
