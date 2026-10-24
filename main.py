"""RAG from scratch - interface de linha de comando.

Uso:
    python main.py build                  # baixa o PDF e cria o índice
    python main.py search "pergunta"      # só a busca semântica
    python main.py ask "pergunta"         # busca + resposta do LLM
    python main.py chat                   # modo interativo
"""
import argparse

from rag import config
from rag.embeddings import build_index, load_embedding_model, load_index
from rag.ingest import chunk_pages, download_pdf, open_and_read_pdf
from rag.retrieval import print_results, retrieve


def cmd_build(_args):
    pdf = download_pdf()
    chunks = chunk_pages(open_and_read_pdf(pdf))
    build_index(chunks, load_embedding_model())


def cmd_search(args):
    chunks, embeddings = load_index()
    print_results(args.query, load_embedding_model(), embeddings, chunks, args.k)


def answer(query, emb_model, embeddings, chunks, generator, k, show_context):
    from rag.generation import prompt_formatter

    scores, indices = retrieve(query, emb_model, embeddings, k)
    context = [chunks[i] for i in indices]
    print(f"\nResposta:\n{generator.generate(prompt_formatter(query, context))}\n")
    if show_context:
        print("Fontes:")
        for s, c in zip(scores, context):
            print(f"  [p. {c['page_number']}] score={s:.3f} {c['sentence_chunk'][:120]}...")


def cmd_ask(args, interactive=False):
    from rag.generation import Generator

    chunks, embeddings = load_index()
    emb_model = load_embedding_model()
    generator = Generator()
    if not interactive:
        answer(args.query, emb_model, embeddings, chunks, generator, args.k, args.sources)
        return
    print("Digite sua pergunta (ou 'sair'):")
    while (q := input("> ").strip()).lower() not in {"sair", "exit", "quit"}:
        if q:
            answer(q, emb_model, embeddings, chunks, generator, args.k, args.sources)


def main():
    parser = argparse.ArgumentParser(description="RAG local do zero")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    for name in ("search", "ask", "chat"):
        p = sub.add_parser(name)
        if name != "chat":
            p.add_argument("query")
        p.add_argument("-k", type=int, default=config.TOP_K)
        p.add_argument("--sources", action="store_true", help="mostra os trechos usados")

    args = parser.parse_args()
    if args.command == "build":
        cmd_build(args)
    elif args.command == "search":
        cmd_search(args)
    else:
        cmd_ask(args, interactive=args.command == "chat")


if __name__ == "__main__":
    main()
