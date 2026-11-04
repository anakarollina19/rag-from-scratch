"""Testes do agente com um LLM falso (não baixam modelos)."""
import unittest

from rag.agent import Agent, parse_output, safe_calc

CHUNK = {"page_number": 7, "sentence_chunk": "Scurvy is caused by a lack of vitamin C.", "score": 0.9}


class FakeRAG:
    chunks = [CHUNK]

    def search(self, query, k):
        return [CHUNK]

    def search_all(self, query, k):
        return [{**CHUNK, "document": "livro"}]


def scripted_llm(outputs):
    """LLM falso que devolve as saídas na ordem e guarda os prompts recebidos."""
    prompts = []

    def llm(prompt, max_new_tokens):
        prompts.append(prompt)
        return outputs[len(prompts) - 1]
    llm.prompts = prompts
    return llm


class ParseOutputTest(unittest.TestCase):
    def test_action(self):
        thought, tool, arg, final = parse_output("Thought: search it\nAction: search[scurvy cause]")
        self.assertEqual((thought, tool, arg, final), ("search it", "search", "scurvy cause", None))

    def test_final(self):
        self.assertEqual(parse_output("Thought: done\nFinal Answer: Vitamin C (p. 7).")[3],
                         "Vitamin C (p. 7).")

    def test_ignores_invented_observation(self):
        text = "Action: search[x]\nObservation: made up\nFinal Answer: wrong"
        self.assertEqual(parse_output(text)[1], "search")


class SafeCalcTest(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(safe_calc("2000 * 0,15"), "300")
        self.assertEqual(safe_calc("2^3 + 1"), "9")

    def test_rejects_code(self):
        self.assertTrue(safe_calc("__import__('os')").startswith("Erro"))


class AgentLoopTest(unittest.TestCase):
    def test_uses_tool_then_answers(self):
        llm = scripted_llm(["Thought: look it up\nAction: search[scurvy]",
                            "Thought: I know the answer\nFinal Answer: Lack of vitamin C (p. 7)."])
        steps = list(Agent(FakeRAG(), llm=llm).run("What causes scurvy?"))
        self.assertEqual([s.kind for s in steps],
                         ["thought", "action", "observation", "thought", "final"])
        self.assertEqual(steps[2].sources, [CHUNK])
        self.assertIn("Observation: (p. 7) Scurvy", llm.prompts[1])
        self.assertEqual(steps[-1].content, "Lack of vitamin C (p. 7).")

    def test_falls_back_to_search_when_off_format(self):
        llm = scripted_llm(["I think it is vitamin C.", "Final Answer: vitamin C"])
        steps = list(Agent(FakeRAG(), llm=llm).run("What causes scurvy?"))
        self.assertEqual(steps[1].tool, "search")
        self.assertEqual(steps[1].content, "What causes scurvy?")

    def test_answers_when_steps_run_out(self):
        llm = scripted_llm(["Action: search_all[scurvy]"] * 2 + ["Vitamin C."])
        steps = list(Agent(FakeRAG(), llm=llm, max_steps=2).run("Scurvy?"))
        self.assertEqual(steps[-1].kind, "final")
        self.assertEqual(steps[-1].content, "Vitamin C.")


if __name__ == "__main__":
    unittest.main()
