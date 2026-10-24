"""Junta as 4 etapas num objeto só, usado pela CLI (main.py) e pela interface web (app.py)."""
from . import config
from .embeddings import build_index, index_path_for, load_embedding_model, load_index
from .ingest import chunk_pages, open_and_read_pdf
from .retrieval import retrieve


class RAG:
    def __init__(self):
        # Os modelos são carregados só quando necessários: o embedding é leve,
        # mas o Gemma ocupa ~10 GB de RAM e leva um tempo para carregar.
        self._embedding_model = None
        self._generator = None
        self.chunks: list[dict] = []
        self.embeddings = None
        self.index_path = ""

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = load_embedding_model()
        return self._embedding_model

    @property
    def generator(self):
        if self._generator is None:
            from .generation import Generator  # importa transformers só quando precisa
            self._generator = Generator()
        return self._generator

    def index_pdf(self, pdf_path: str, page_offset: int = 0) -> str:
        """Lê, divide e indexa um PDF; o índice vira o ativo. Retorna o caminho do índice."""
        # str(): o Gradio entrega o caminho como uma subclasse de str, que o
        # torch.load (com weights_only=True) se recusa a ler de volta
        pdf_path = str(pdf_path)
        chunks = chunk_pages(open_and_read_pdf(pdf_path, page_offset))
        path = index_path_for(pdf_path)
        build_index(chunks, self.embedding_model, path, source=pdf_path)
        self.load(path)
        return path

    def load(self, index_path: str) -> None:
        self.chunks, self.embeddings = load_index(index_path)
        self.index_path = index_path

    def search(self, query: str, k: int = config.TOP_K) -> list[dict]:
        """Retorna os k chunks mais parecidos, cada um com seu score."""
        if self.embeddings is None:
            raise RuntimeError("Nenhum índice carregado. Rode `build` ou envie um PDF.")
        scores, indices = retrieve(query, self.embedding_model, self.embeddings, k)
        return [{**self.chunks[i], "score": float(s)} for s, i in zip(scores, indices)]

    def ask(self, query: str, k: int = config.TOP_K) -> tuple[str, list[dict]]:
        """Busca + geração. Retorna (resposta, trechos usados)."""
        from .generation import prompt_formatter

        context = self.search(query, k)
        return self.generator.generate(prompt_formatter(query, context)), context
