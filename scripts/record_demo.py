"""Grava o GIF da interface usado no README (docs/demo.gif).

Pré-requisitos:
    pip install playwright && python -m playwright install chromium
    python app.py            # em outro terminal, com o livro de exemplo indexado

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
WIDTH, HEIGHT = 1200, 820
GIF_WIDTH = 900


def main(question: str):
    frames: list[tuple[Image.Image, int]] = []  # (imagem, duração em ms)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})

        def snap(ms: int):
            frames.append((Image.open(io.BytesIO(page.screenshot())).convert("RGB"), ms))

        page.goto(URL)
        page.get_by_text("Documento ativo").wait_for()
        snap(1500)

        box = page.get_by_label("Pergunta")
        box.click()
        for i in range(0, len(question), 4):  # "digita" aos poucos
            box.fill(question[: i + 4])
            snap(120)
        snap(800)

        page.get_by_role("button", name="Perguntar").click()
        page.wait_for_timeout(1500)
        snap(1500)  # carregando (o tempo real de espera não entra no GIF)

        page.get_by_text("Página", exact=False).first.wait_for(timeout=300_000)
        page.wait_for_timeout(500)
        snap(4000)  # resposta

        page.mouse.wheel(0, 500)  # rola até as fontes
        page.wait_for_timeout(500)
        snap(4000)
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
