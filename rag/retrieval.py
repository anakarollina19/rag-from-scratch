"""Etapa 3: busca semântica por similaridade e re-ranking opcional."""
import torch
from sentence_transformers import CrossEncoder, util

from . import config
from .embeddings import Embedder


def retrieve(query: str, embedder: Embedder, embeddings: torch.Tensor,
             k: int = config.TOP_K) -> tuple[torch.Tensor, torch.Tensor]:
    """Retorna (scores, índices) dos k chunks mais parecidos com a pergunta."""
    query_emb = embedder.encode_query(query).to(embeddings.device)
    scores = util.dot_score(query_emb, embeddings)[0]
    return torch.topk(scores, k=min(k, len(scores)))  # PDFs pequenos podem ter < k chunks


class Reranker:
    """Cross-encoder: lê pergunta e trecho *juntos* e dá uma nota de relevância.

    É mais preciso que comparar dois vetores calculados separadamente, mas bem
    mais lento, por isso só é aplicado aos poucos candidatos que a busca trouxe.
    """

    def __init__(self, model_name: str = config.RERANKER_MODEL):
        self.model = CrossEncoder(model_name, device=config.DEVICE)

    def rerank(self, query: str, chunks: list[dict], k: int) -> list[dict]:
        scores = self.model.predict([(query, c["sentence_chunk"]) for c in chunks])
        ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
        return [{**c, "rerank_score": float(s)} for s, c in ranked[:k]]
