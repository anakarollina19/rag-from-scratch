"""RAG from scratch - interface de linha de comando.

Uso:
    python main.py build                  # baixa o livro de exemplo e cria o índice
    python main.py build meu.pdf          # indexa qualquer PDF
    python main.py search "pergunta"      # só a busca semântica
    python main.py ask "pergunta"         # busca + resposta do LLM
    python main.py chat                   # modo interativo

search/ask/chat usam o índice mais recente; para escolher outro, use --index.
"""
import argparse
import sys

from rag import config
from rag.embeddings import list_indexes
from rag.ingest import download_pdf
from rag.pipeline import RAG


def cmd_build(args, rag: RAG):
    if args.pdf:
        rag.index_pdf(args.pdf, page_offset=args.page_offset)
    else:
        rag.index_pdf(download_pdf(), page_offset=config.BOOK_PAGE_OFFSET)


def load_index_or_exit(args, rag: RAG):
    path = args.index or next(iter(list_indexes()), None)
    if path is None:
        sys.exit("Nenhum índice encontrado. Rode primeiro: python main.py build [arquivo.pdf]")
    rag.load(path)
    print(f"[INFO] Usando o índice {path}")


def print_sources(chunks: list[dict], full_text: bool = False):
    for c in chunks:
        text = c["sentence_chunk"] if full_text else c["sentence_chunk"][:120] + "..."
        rerank = f" rerank={c['rerank_score']:.2f}" if "rerank_score" in c else ""
        print(f"  [p. {c['page_number']}] score={c['score']:.3f}{rerank} {text}")


def cmd_search(args, rag: RAG):
    print(f"Pergunta: {args.query}\n")
    print_sources(rag.search(args.query, args.k), full_text=True)


def answer(rag: RAG, query: str, k: int, show_sources: bool):
    text, context = rag.ask(query, k)
    print(f"\nResposta:\n{text}\n")
    if show_sources:
        print("Fontes:")
        print_sources(context)


def cmd_chat(args, rag: RAG):
    print("Digite sua pergunta (ou 'sair'):")
    while (q := input("> ").strip()).lower() not in {"sair", "exit", "quit"}:
        if q:
            answer(rag, q, args.k, args.sources)


def main():
    parser = argparse.ArgumentParser(description="RAG local do zero")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="indexa um PDF")
    build.add_argument("pdf", nargs="?", help="caminho do PDF (padrão: livro de exemplo)")
    build.add_argument("--page-offset", type=int, default=0,
                       help="páginas iniciais a descontar na numeração (capa, sumário...)")
    build.add_argument("--model", default=config.EMBEDDING_MODEL,
                       choices=list(config.EMBEDDING_MODELS), help="modelo de embedding")

    for name in ("search", "ask", "chat"):
        p = sub.add_parser(name)
        if name != "chat":
            p.add_argument("query")
        p.add_argument("-k", type=int, default=config.TOP_K)
        p.add_argument("--index", help="arquivo .pt em data/index (padrão: o mais recente)")
        p.add_argument("--sources", action="store_true", help="mostra os trechos usados")
        p.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=config.RERANK,
                       help="reordena os resultados com um cross-encoder")

    args = parser.parse_args()
    rag = RAG(embedding_model=getattr(args, "model", config.EMBEDDING_MODEL),
              rerank=getattr(args, "rerank", config.RERANK))
    if args.command == "build":
        cmd_build(args, rag)
        return

    load_index_or_exit(args, rag)
    if args.command == "search":
        cmd_search(args, rag)
    elif args.command == "ask":
        answer(rag, args.query, args.k, args.sources)
    else:
        cmd_chat(args, rag)


if __name__ == "__main__":
    main()
