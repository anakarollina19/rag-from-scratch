"""Junta as 4 etapas num objeto só, usado pela CLI (main.py) e pela interface web (app.py)."""
from . import config
from .embeddings import Embedder, build_index, index_path_for, load_index
from .ingest import chunk_pages, open_and_read_pdf
from .retrieval import Reranker, retrieve


class RAG:
    def __init__(self, embedding_model: str = config.EMBEDDING_MODEL,
                 rerank: bool = config.RERANK):
        # Os modelos são carregados só quando necessários: o embedding é leve,
        # mas o Gemma ocupa ~10 GB de RAM e leva um tempo para carregar.
        self.embedding_model = embedding_model
        self.rerank = rerank
        self._embedders: dict[str, Embedder] = {}
        self._reranker = None
        self._generator = None
        self.chunks: list[dict] = []
        self.embeddings = None
        self.index_path = ""
        self.index_model = ""  # modelo com que o índice ativo foi construído

    def embedder(self, model_name: str) -> Embedder:
        if model_name not in self._embedders:
            self._embedders[model_name] = Embedder(model_name)
        return self._embedders[model_name]

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    @property
    def generator(self):
        if self._generator is None:
            from .generation import Generator  # importa transformers só quando precisa
            self._generator = Generator()
        return self._generator

    def index_pdf(self, pdf_path: str, page_offset: int = 0,
                  first_page: int = 1, last_page: int | None = None) -> str:
        """Lê, divide e indexa um PDF (ou só um intervalo de páginas dele).

        O índice vira o ativo. Retorna o caminho do índice.
        """
        # str(): o Gradio entrega o caminho como uma subclasse de str, que o
        # torch.load (com weights_only=True) se recusa a ler de volta
        pdf_path = str(pdf_path)
        pages = open_and_read_pdf(pdf_path, page_offset, first_page, last_page)
        chunks = chunk_pages(pages)
        path = index_path_for(pdf_path)
        if first_page > 1 or last_page:  # um índice por intervalo, sem sobrescrever o PDF inteiro
            path = path.replace(".pt", f"_p{first_page}-{last_page or 'fim'}.pt")
        build_index(chunks, self.embedder(self.embedding_model), path, source=pdf_path)
        self.load(path)
        return path

    def load(self, index_path: str) -> None:
        self.chunks, self.embeddings, self.index_model = load_index(index_path)
        self.index_path = index_path

    def search(self, query: str, k: int = config.TOP_K) -> list[dict]:
        """Retorna os k chunks mais relevantes, cada um com seu score."""
        if self.embeddings is None:
            raise RuntimeError("Nenhum índice carregado. Rode `build` ou envie um PDF.")
        # a pergunta precisa ser codificada com o mesmo modelo que gerou o índice
        n = max(k, config.RERANK_CANDIDATES) if self.rerank else k
        scores, indices = retrieve(query, self.embedder(self.index_model), self.embeddings, n)
        found = [{**self.chunks[i], "score": float(s)} for s, i in zip(scores, indices)]
        return self.reranker.rerank(query, found, k) if self.rerank else found

    def ask(self, query: str, k: int = config.TOP_K) -> tuple[str, list[dict]]:
        """Busca + geração. Retorna (resposta, trechos usados)."""
        from .generation import prompt_formatter

        context = self.search(query, k)
        return self.generator.generate(prompt_formatter(query, context)), context
