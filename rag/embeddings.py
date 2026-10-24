"""Etapa 2: gerar, salvar e carregar embeddings dos chunks."""
import os

import torch
from sentence_transformers import SentenceTransformer

from . import config


def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL, device=config.DEVICE)


def build_index(chunks: list[dict], model: SentenceTransformer,
                path: str = config.EMBEDDINGS_PATH) -> torch.Tensor:
    """Gera embeddings em lote e salva chunks + tensores em um único arquivo."""
    texts = [c["sentence_chunk"] for c in chunks]
    embeddings = model.encode(texts, batch_size=32, convert_to_tensor=True,
                              show_progress_bar=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"chunks": chunks, "embeddings": embeddings.cpu()}, path)
    print(f"[INFO] Índice salvo em {path} ({len(chunks)} chunks)")
    return embeddings


def load_index(path: str = config.EMBEDDINGS_PATH) -> tuple[list[dict], torch.Tensor]:
    data = torch.load(path)
    return data["chunks"], data["embeddings"].to(config.DEVICE)
