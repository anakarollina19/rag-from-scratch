"""Avalia a qualidade da BUSCA (não da geração) no livro de exemplo.

Para cada pergunta de eval/questions.json sabemos em quais páginas está a
resposta. Medimos se a busca traz essas páginas:

  hit@3  -> % de perguntas com pelo menos uma página certa entre os 3 primeiros
            trechos (os 3 que vão para o Gemma, TOP_K=3)
  MRR@10 -> média de 1/posição do primeiro trecho certo (1.0 = sempre em 1º;
            0 = não apareceu entre os 10 primeiros)

Uso (rode `python main.py build` antes, para ter o livro indexado):
    python -m eval.run_eval
    python -m eval.run_eval --models all-mpnet-base-v2 --langs en --no-rerank
"""
import argparse
import json
import os
import time

import torch

from rag import config
from rag.embeddings import Embedder, load_index
from rag.retrieval import Reranker, retrieve

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")
BOOK_INDEX = "data/index/human-nutrition-text.pt"
CACHE_DIR = "data/eval_cache"  # embeddings do livro para cada modelo testado


def book_embeddings(chunks: list[dict], embedder: Embedder) -> torch.Tensor:
    """Embeddings do livro com o modelo pedido; calcula uma vez e guarda em cache."""
    path = f"{CACHE_DIR}/{embedder.name.replace('/', '__')}.pt"
    if os.path.exists(path):
        return torch.load(path).to(config.DEVICE)
    print(f"[INFO] Gerando embeddings do livro com {embedder.name} (só na primeira vez)...")
    embeddings = embedder.encode_passages([c["sentence_chunk"] for c in chunks])
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(embeddings.cpu(), path)
    return embeddings


def ranked_pages(query, chunks, embedder, embeddings, reranker, depth=10) -> list[int]:
    n = config.RERANK_CANDIDATES if reranker else depth
    _, indices = retrieve(query, embedder, embeddings, n)
    found = [chunks[i] for i in indices]
    if reranker:
        found = reranker.rerank(query, found, depth)
    return [c["page_number"] for c in found[:depth]]


def score(pages: list[int], relevant: set[int]) -> tuple[float, float]:
    """(hit@3, reciprocal rank@10) de uma pergunta."""
    hit = float(any(p in relevant for p in pages[:3]))
    rank = next((i + 1 for i, p in enumerate(pages) if p in relevant), None)
    return hit, (1 / rank if rank else 0.0)


def main():
    parser = argparse.ArgumentParser(description="Avaliação da busca do RAG")
    parser.add_argument("--models", nargs="+", default=list(config.EMBEDDING_MODELS))
    parser.add_argument("--langs", nargs="+", default=["en", "pt"], choices=["en", "pt"])
    parser.add_argument("--no-rerank", action="store_true", help="não testa com re-ranking")
    parser.add_argument("-v", "--verbose", action="store_true", help="mostra cada pergunta")
    args = parser.parse_args()

    if not os.path.exists(BOOK_INDEX):
        raise SystemExit("Livro não indexado. Rode antes: python main.py build")
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    chunks, _, _ = load_index(BOOK_INDEX)  # reaproveita os chunks; embeddings são por modelo

    reranker = None if args.no_rerank else Reranker()
    rerank_options = [False] if args.no_rerank else [False, True]

    rows = []
    for model_name in args.models:
        embedder = Embedder(model_name)
        embeddings = book_embeddings(chunks, embedder)
        for use_rerank in rerank_options:
            for lang in args.langs:
                start = time.time()
                hits, rrs = [], []
                for q in questions:
                    pages = ranked_pages(q[lang], chunks, embedder, embeddings,
                                         reranker if use_rerank else None)
                    hit, rr = score(pages, set(q["pages"]))
                    hits.append(hit)
                    rrs.append(rr)
                    if args.verbose:
                        print(f"  {'OK ' if hit else 'ERR'} {q[lang]} -> {pages[:3]} (certas: {q['pages']})")
                ms = (time.time() - start) / len(questions) * 1000
                rows.append((model_name, "sim" if use_rerank else "não", lang,
                             sum(hits) / len(hits), sum(rrs) / len(rrs), ms))

    print(f"\n{len(questions)} perguntas\n")
    print("| Modelo de embedding | Re-ranking | Idioma | hit@3 | MRR@10 | ms/pergunta |")
    print("|---|---|---|---|---|---|")
    for model_name, rerank, lang, hit, mrr, ms in rows:
        print(f"| {model_name} | {rerank} | {lang} | {hit:.0%} | {mrr:.2f} | {ms:.0f} |")


if __name__ == "__main__":
    main()
