"""Testes da análise exploratória com dados sintéticos."""
import unittest

import torch

from rag.analysis import best_k, cluster_names, describe, kmeans, pca_2d, silhouette, top_terms


def blobs():
    """3 grupos bem separados em 10 dimensões."""
    g = torch.Generator().manual_seed(0)
    centers = torch.eye(10)[:3] * 10
    return torch.cat([c + torch.randn(20, 10, generator=g) * 0.3 for c in centers])


class AnalysisTest(unittest.TestCase):
    def test_kmeans_finds_groups(self):
        labels, _ = kmeans(blobs(), 3)
        for start in (0, 20, 40):  # cada bloco de 20 pontos cai num só grupo
            self.assertEqual(len(labels[start:start + 20].unique()), 1)
        self.assertEqual(len(labels.unique()), 3)

    def test_best_k_is_three(self):
        self.assertEqual(best_k(blobs(), range(2, 6))[0], 3)

    def test_silhouette_matches_definition(self):
        # 4 pontos na reta: grupos {0, 1} e {10, 11}
        x = torch.tensor([[0.], [1.], [10.], [11.]])
        # ponto 0: a = 1, b = média(10, 11) = 10,5 -> s = 9,5/10,5 (os outros são simétricos)
        expected = ((9.5 / 10.5) + (8.5 / 9.5)) / 2
        self.assertAlmostEqual(silhouette(x, torch.tensor([0, 0, 1, 1])), expected, places=5)

    def test_pca_shape_and_variance(self):
        points, variance = pca_2d(blobs())
        self.assertEqual(points.shape, (60, 2))
        self.assertGreater(sum(variance), 0.9)

    def test_text_stats(self):
        chunks = [{"page_number": 1, "chunk_word_count": 4, "sentence_chunk": "vitamin vitamin bone"},
                  {"page_number": 2, "chunk_word_count": 6, "sentence_chunk": "the protein bone"}]
        self.assertEqual(describe(chunks)["Trechos"], 2)
        self.assertEqual(top_terms(chunks)[0], ("vitamin", 2))
        names = cluster_names(chunks, torch.tensor([0, 1]), n_terms=1)
        self.assertEqual(names, {0: "vitamin", 1: "protein"})


if __name__ == "__main__":
    unittest.main()
