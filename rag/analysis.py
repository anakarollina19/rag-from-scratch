"""Análise exploratória do documento indexado (ciência de dados "de base").

Tudo feito com PyTorch, sem scikit-learn, como o resto do projeto:
- estatísticas descritivas dos trechos;
- termos mais frequentes e TF-IDF;
- PCA para projetar os embeddings (768 dimensões) em 2D;
- K-means para agrupar os trechos por tema, com o coeficiente de silhueta
  para escolher o número de grupos.
"""
import math
import re
from collections import Counter

import torch

STOPWORDS = set("""
a about above after again all also am an and any are as at be because been before being
between both but by can could did do does doing down during each few for from further had
has have having he her here hers him his how i if in into is it its itself just may me
might more most must my no nor not of off on once only or other our out over own same she
should so some such than that the their them then there these they this those through to
too under until up very was we were what when where which while who whom why will with
would you your one two also used use many much however e.g i.e figure table chapter
o os as um uma uns umas de do da dos das em no na nos nas por para com sem que se não
mais muito como ao aos à às é são ser foi pelo pela pelos pelas seu sua seus suas ou
mas também entre sobre este esta isso isto essa esse ele ela eles elas já quando
""".split())

WORD = re.compile(r"[a-zà-ú]{3,}", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [w for w in (m.lower() for m in WORD.findall(text)) if w not in STOPWORDS]


def describe(chunks: list[dict]) -> dict:
    """Estatísticas descritivas do tamanho dos trechos (em palavras)."""
    words = torch.tensor([float(c["chunk_word_count"]) for c in chunks])
    pages = {c["page_number"] for c in chunks}
    return {
        "Trechos": len(chunks),
        "Páginas com texto": len(pages),
        "Palavras (total)": int(words.sum()),
        "Palavras por trecho (média)": round(words.mean().item(), 1),
        "Desvio padrão": round(words.std().item(), 1) if len(words) > 1 else 0.0,
        "Mediana": round(words.median().item(), 1),
        "Mínimo / máximo": f"{int(words.min())} / {int(words.max())}",
    }


def top_terms(chunks: list[dict], n: int = 15) -> list[tuple[str, int]]:
    counts = Counter(w for c in chunks for w in tokenize(c["sentence_chunk"]))
    return counts.most_common(n)


def pca_2d(embeddings: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
    """Projeta em 2 componentes principais. Retorna (pontos, variância explicada)."""
    x = embeddings.float().cpu()
    x = x - x.mean(dim=0)
    # SVD da matriz centralizada: as linhas de Vh são as direções principais
    _, s, vh = torch.linalg.svd(x, full_matrices=False)
    variance = (s ** 2) / (s ** 2).sum()
    return x @ vh[:2].T, variance[:2].tolist()


def kmeans(x: torch.Tensor, k: int, iters: int = 50, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """K-means com inicialização k-means++. Retorna (rótulos, centróides)."""
    x = x.float().cpu()
    g = torch.Generator().manual_seed(seed)
    centroids = [x[torch.randint(len(x), (1,), generator=g)].squeeze(0)]
    for _ in range(1, k):  # k-means++: novos centros longe dos já escolhidos
        dist = torch.cdist(x, torch.stack(centroids)).min(dim=1).values ** 2
        centroids.append(x[torch.multinomial(dist / dist.sum(), 1, generator=g)].squeeze(0))
    centroids = torch.stack(centroids)

    for _ in range(iters):
        labels = torch.cdist(x, centroids).argmin(dim=1)
        new = torch.stack([x[labels == j].mean(dim=0) if (labels == j).any() else centroids[j]
                           for j in range(k)])
        if torch.allclose(new, centroids, atol=1e-5):
            break
        centroids = new
    return labels, centroids


def silhouette(x: torch.Tensor, labels: torch.Tensor, max_points: int = 1500) -> float:
    """Coeficiente de silhueta médio (−1 a 1): quanto maior, mais separados os grupos."""
    x, labels = x.float().cpu(), labels.cpu()
    if len(x) > max_points:  # amostra para não montar uma matriz gigante
        idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))[:max_points]
        x, labels = x[idx], labels[idx]
    d = torch.cdist(x, x)
    scores = []
    for i in range(len(x)):
        same = labels == labels[i]
        if same.sum() <= 1:
            continue
        a = d[i][same].sum() / (same.sum() - 1)
        b = min(d[i][labels == j].mean() for j in labels.unique() if j != labels[i])
        scores.append(((b - a) / max(a, b)).item())
    return sum(scores) / len(scores) if scores else 0.0


def best_k(x: torch.Tensor, k_range=range(2, 9)) -> tuple[int, dict[int, float]]:
    """Escolhe o k com maior silhueta."""
    scores = {k: silhouette(x, kmeans(x, k)[0]) for k in k_range if k < len(x)}
    return max(scores, key=scores.get), scores


def cluster_names(chunks: list[dict], labels: torch.Tensor, n_terms: int = 3) -> dict[int, str]:
    """Nomeia cada grupo pelos termos com maior TF-IDF (termo frequente no grupo, raro nos outros)."""
    groups: dict[int, Counter] = {}
    for c, label in zip(chunks, labels.tolist()):
        groups.setdefault(label, Counter()).update(tokenize(c["sentence_chunk"]))
    doc_freq = Counter(w for counts in groups.values() for w in counts)
    names = {}
    for label, counts in groups.items():
        total = sum(counts.values()) or 1
        score = {w: (n / total) * math.log(1 + len(groups) / doc_freq[w]) for w, n in counts.items()}
        names[label] = ", ".join(sorted(score, key=score.get, reverse=True)[:n_terms])
    return names


def analyze(chunks: list[dict], embeddings: torch.Tensor, k: int | None = None) -> dict:
    """Roda toda a análise de uma vez (usada pela aba Análise da interface)."""
    points, variance = pca_2d(embeddings)
    if k is None:
        k, scores = best_k(embeddings)
    else:
        scores = {k: None}
    labels, _ = kmeans(embeddings, k)
    names = cluster_names(chunks, labels)
    return {"stats": describe(chunks), "terms": top_terms(chunks), "points": points,
            "variance": variance, "labels": labels, "names": names, "k": k,
            "silhouette": scores}
