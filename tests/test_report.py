"""Testes do relatório exportado e da leitura por intervalo de páginas. Rode com: python -m unittest"""
import os
import tempfile
import unittest

import pymupdf

from rag.ingest import open_and_read_pdf
from rag.pdf_preview import page_count, thumbnails
from rag.report import build_html, html_to_pdf

ENTRY = {"question": "O que é <b>colostro</b>?", "answer": "O primeiro leite.",
         "sources": [{"page_number": 827, "sentence_chunk": "Colostrum is..."}],
         "document": "livro", "time": "10:00"}


def make_pdf(path: str, pages: int):
    doc = pymupdf.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"Texto da pagina {i + 1}. " * 5)
    doc.save(path)


class ReportTest(unittest.TestCase):
    def test_html_escapes_question_and_lists_sources(self):
        page = build_html([ENTRY])
        self.assertIn("O que é &lt;b&gt;colostro&lt;/b&gt;?", page)
        self.assertIn("p. 827", page)

    def test_html_without_sources(self):
        self.assertNotIn("p. 827", build_html([ENTRY], include_sources=False))

    def test_pdf_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = html_to_pdf(build_html([ENTRY] * 3), os.path.join(tmp, "r.pdf"))
            with pymupdf.open(path) as doc:
                self.assertIn("colostro", doc[0].get_text())


class PageRangeTest(unittest.TestCase):
    def test_reads_only_requested_pages_with_real_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "doc.pdf")
            make_pdf(path, 5)
            pages = open_and_read_pdf(path, first_page=2, last_page=3)
            self.assertEqual([p["page_number"] for p in pages], [2, 3])
            self.assertEqual(page_count(path), 5)
            self.assertEqual([c for _, c in thumbnails(path, 4)], ["Página 4", "Página 5"])


if __name__ == "__main__":
    unittest.main()
