"""Etapa 2: gerar, salvar e carregar embeddings dos chunks."""
import os
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from . import config


def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL, device=config.DEVICE)


def index_path_for(pdf_path: str) -> str:
    """data/index/<nome-do-pdf>.pt"""
    return f"{config.INDEX_DIR}/{Path(pdf_path).stem}.pt"  # "/" funciona no Windows também


def build_index(chunks: list[dict], model: SentenceTransformer, path: str,
                source: str = "") -> torch.Tensor:
    """Gera embeddings em lote e salva chunks + tensores em um único arquivo."""
    if not chunks:
        raise ValueError("Nenhum texto encontrado no PDF (ele pode ser só imagens escaneadas).")
    texts = [c["sentence_chunk"] for c in chunks]
    embeddings = model.encode(texts, batch_size=32, convert_to_tensor=True,
                              show_progress_bar=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"source": source, "chunks": chunks, "embeddings": embeddings.cpu()}, path)
    print(f"[INFO] Índice salvo em {path} ({len(chunks)} chunks)")
    return embeddings


def load_index(path: str) -> tuple[list[dict], torch.Tensor]:
    data = torch.load(path)
    return data["chunks"], data["embeddings"].to(config.DEVICE)


def list_indexes() -> list[str]:
    """Índices disponíveis, do mais recente para o mais antigo."""
    if not os.path.isdir(config.INDEX_DIR):
        return []
    paths = [f"{config.INDEX_DIR}/{f}" for f in os.listdir(config.INDEX_DIR)
             if f.endswith(".pt")]
    return sorted(paths, key=os.path.getmtime, reverse=True)
