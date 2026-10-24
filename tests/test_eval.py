"""Testes das métricas de avaliação. Rode com: python -m unittest"""
import json
import unittest

from eval.run_eval import QUESTIONS_PATH, score


class ScoreTest(unittest.TestCase):
    def test_first_position(self):
        self.assertEqual(score([10, 20, 30], {10}), (1.0, 1.0))

    def test_third_position_is_still_a_hit(self):
        self.assertEqual(score([1, 2, 10], {10}), (1.0, 1 / 3))

    def test_found_after_top3_is_not_a_hit(self):
        hit, rr = score([1, 2, 3, 10], {10})
        self.assertEqual(hit, 0.0)
        self.assertAlmostEqual(rr, 0.25)

    def test_not_found(self):
        self.assertEqual(score([1, 2, 3], {99}), (0.0, 0.0))


class QuestionsFileTest(unittest.TestCase):
    def test_every_question_has_both_languages_and_pages(self):
        with open(QUESTIONS_PATH, encoding="utf-8") as f:
            questions = json.load(f)
        self.assertGreaterEqual(len(questions), 20)
        for q in questions:
            self.assertTrue(q["en"] and q["pt"] and q["pages"], q)


if __name__ == "__main__":
    unittest.main()
