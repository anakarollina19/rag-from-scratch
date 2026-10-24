"""Etapa 2: gerar, salvar e carregar embeddings dos chunks."""
import os
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from . import config


class Embedder:
    """Modelo de embedding + os prefixos que ele espera (ver config.EMBEDDING_MODELS)."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.name = model_name
        prefixes = config.EMBEDDING_MODELS.get(model_name, {})
        self.query_prefix = prefixes.get("query_prefix", "")
        self.passage_prefix = prefixes.get("passage_prefix", "")
        self.model = SentenceTransformer(model_name, device=config.DEVICE)

    def encode_passages(self, texts: list[str], show_progress: bool = True) -> torch.Tensor:
        return self.model.encode([self.passage_prefix + t for t in texts], batch_size=32,
                                 convert_to_tensor=True, normalize_embeddings=True,
                                 show_progress_bar=show_progress)

    def encode_query(self, query: str) -> torch.Tensor:
        # vetores normalizados: o produto escalar vira similaridade de cosseno
        return self.model.encode(self.query_prefix + query, convert_to_tensor=True,
                                 normalize_embeddings=True)


def index_path_for(pdf_path: str) -> str:
    """data/index/<nome-do-pdf>.pt"""
    return f"{config.INDEX_DIR}/{Path(pdf_path).stem}.pt"  # "/" funciona no Windows também


def build_index(chunks: list[dict], embedder: Embedder, path: str,
                source: str = "") -> torch.Tensor:
    """Gera embeddings em lote e salva chunks + tensores em um único arquivo."""
    if not chunks:
        raise ValueError("Nenhum texto encontrado no PDF (ele pode ser só imagens escaneadas).")
    embeddings = embedder.encode_passages([c["sentence_chunk"] for c in chunks])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"source": source, "embedding_model": embedder.name,
                "chunks": chunks, "embeddings": embeddings.cpu()}, path)
    print(f"[INFO] Índice salvo em {path} ({len(chunks)} chunks, {embedder.name})")
    return embeddings


def load_index(path: str) -> tuple[list[dict], torch.Tensor, str]:
    """Retorna (chunks, embeddings, nome do modelo de embedding usado no índice)."""
    data = torch.load(path)
    # índices antigos não guardavam o modelo; eles foram feitos com o all-mpnet-base-v2
    model_name = data.get("embedding_model", "all-mpnet-base-v2")
    return data["chunks"], data["embeddings"].to(config.DEVICE), model_name


def list_indexes() -> list[str]:
    """Índices disponíveis, do mais recente para o mais antigo."""
    if not os.path.isdir(config.INDEX_DIR):
        return []
    paths = [f"{config.INDEX_DIR}/{f}" for f in os.listdir(config.INDEX_DIR)
             if f.endswith(".pt")]
    return sorted(paths, key=os.path.getmtime, reverse=True)
