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
    return torch.topk(scores, k=k)


def print_results(query: str, model, embeddings, chunks, k: int = config.TOP_K) -> None:
    scores, indices = retrieve(query, model, embeddings, k)
    print(f"Pergunta: {query}\n")
    for score, idx in zip(scores, indices):
        chunk = chunks[idx]
        print(f"Score: {score:.4f} | Página: {chunk['page_number']}")
        print(chunk["sentence_chunk"], "\n")
