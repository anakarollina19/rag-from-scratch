"""Grava o GIF da interface usado no README (docs/demo.gif).

Pré-requisitos:
    pip install playwright && python -m playwright install chromium
    python main.py build     # o livro de exemplo precisa estar indexado
    python app.py            # em outro terminal

Uso:
    python scripts/record_demo.py "How does saliva help with digestion?"
"""
import io
import os
import sys

from PIL import Image
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:7860"
OUT = "docs/demo.gif"
WIDTH, HEIGHT = 1280, 860
GIF_WIDTH = 900
BOOK_PDF = "data/human-nutrition-text.pdf"
BOOK_INDEX_NAME = "human-nutrition-text"


def main(question: str):
    frames: list[tuple[Image.Image, int]] = []  # (imagem, duração em ms)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, color_scheme="dark")

        def snap(ms: int):
            frames.append((Image.open(io.BytesIO(page.screenshot())).convert("RGB"), ms))

        page.goto(URL)
        page.get_by_text("trechos indexados").wait_for()

        # 1. Documentos: envio do PDF, prévia e escolha das páginas
        page.get_by_role("tab", name="📄 Documentos").click()
        page.locator("input[type=file]").set_input_files(BOOK_PDF)
        page.get_by_text("Página 12").wait_for(timeout=60_000)
        page.wait_for_timeout(1000)
        snap(2500)

        # 2. Perguntar: usa o livro inteiro, já indexado
        page.get_by_role("tab", name="💬 Perguntar").click()
        page.get_by_label("Trocar documento").click()
        page.get_by_role("option", name=BOOK_INDEX_NAME, exact=True).click()
        page.wait_for_timeout(1000)
        snap(1500)

        box = page.get_by_placeholder("Digite sua pergunta")
        box.click()
        for i in range(0, len(question), 4):  # "digita" aos poucos
            box.fill(question[: i + 4])
            snap(120)
        box.press("Enter")
        page.get_by_text("Gerando a resposta").wait_for(timeout=60_000)
        snap(1800)  # o tempo real de espera não entra no GIF

        page.get_by_text("Gerando a resposta").wait_for(state="detached", timeout=300_000)
        page.wait_for_timeout(500)
        snap(4000)  # resposta

        page.get_by_text("Fontes · ", exact=False).click()  # abre as fontes
        page.wait_for_timeout(600)
        snap(3500)

        # 3. Exportar: prévia do relatório
        page.get_by_role("tab", name="🖨️ Exportar").click()
        page.wait_for_timeout(1500)
        snap(3500)
        browser.close()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    scale = GIF_WIDTH / WIDTH
    images = [img.resize((GIF_WIDTH, int(img.height * scale)), Image.LANCZOS)
              .quantize(colors=128, method=Image.Quantize.MEDIANCUT) for img, _ in frames]
    images[0].save(OUT, save_all=True, append_images=images[1:],
                   duration=[ms for _, ms in frames], loop=0, optimize=True)
    print(f"GIF salvo em {OUT} ({len(images)} quadros, {os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "How does saliva help with digestion?")
