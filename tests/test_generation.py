"""Testes rápidos (não baixam modelos). Rode com: python -m unittest"""
import unittest

from rag.generation import clean_answer, prompt_formatter


class CleanAnswerTest(unittest.TestCase):
    def test_removes_sure_heres(self):
        text = "Sure, here's the answer to the question:\n\nThe macronutrients are carbs."
        self.assertEqual(clean_answer(text), "The macronutrients are carbs.")

    def test_removes_according_to_context(self):
        text = "According to the context, saliva helps with digestion."
        self.assertEqual(clean_answer(text), "Saliva helps with digestion.")

    def test_removes_certainly_here_is(self):
        self.assertEqual(clean_answer("Certainly! Here is the answer: Water matters."),
                         "Water matters.")

    def test_keeps_normal_answer(self):
        text = "The context explains that fiber plays a role."
        self.assertEqual(clean_answer(text), text)


class PromptFormatterTest(unittest.TestCase):
    def test_includes_context_and_query(self):
        prompt = prompt_formatter("What is fiber?", [{"sentence_chunk": "Fiber is a carb."}])
        self.assertIn("- Fiber is a carb.", prompt)
        self.assertIn("Question: What is fiber?", prompt)


if __name__ == "__main__":
    unittest.main()
