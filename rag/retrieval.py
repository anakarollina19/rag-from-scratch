"""Etapa 3: busca semântica por similaridade (produto escalar)."""
import torch
from sentence_transformers import SentenceTransformer, util

from . import config


def retrieve(query: str, model: SentenceTransformer, embeddings: torch.Tensor,
             k: int = config.TOP_K) -> tuple[torch.Tensor, torch.Tensor]:
    """Retorna (scores, índices) dos k chunks mais parecidos com a pergunta."""
    query_emb = model.encode(query, convert_to_tensor=True).to(embeddings.device)
    # all-mpnet-base-v2 gera vetores normalizados, então dot product == cosseno
    scores = util.dot_score(query_emb, embeddings)[0]
    return torch.topk(scores, k=min(k, len(scores)))  # PDFs pequenos podem ter < k chunks
