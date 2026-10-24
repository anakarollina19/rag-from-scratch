"""Miniaturas das páginas de um PDF, para conferir o arquivo antes de indexar."""
import pymupdf
from PIL import Image


def page_count(pdf_path: str) -> int:
    with pymupdf.open(pdf_path) as doc:
        return len(doc)


def thumbnails(pdf_path: str, first_page: int = 1, last_page: int | None = None,
               limit: int = 24, width: int = 220) -> list[tuple[Image.Image, str]]:
    """(imagem, legenda) das páginas do intervalo; no máximo `limit` páginas."""
    with pymupdf.open(pdf_path) as doc:
        last_page = min(last_page or len(doc), len(doc))
        images = []
        for number in range(first_page, min(last_page, first_page + limit - 1) + 1):
            page = doc[number - 1]
            pix = page.get_pixmap(matrix=pymupdf.Matrix(width / page.rect.width, width / page.rect.width))
            images.append((Image.frombytes("RGB", (pix.width, pix.height), pix.samples),
                           f"Página {number}"))
        return images
