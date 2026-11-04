"""Agente ReAct: o LLM decide, passo a passo, qual ferramenta usar.

Em vez de sempre "buscar 3 trechos e responder", o modelo alterna entre
pensar (Thought), agir (Action: ferramenta[entrada]) e ler o resultado
(Observation), até dar a resposta final (Final Answer).

Feito do zero, sem LangChain: o "protocolo" é só texto, lido com regex.
Modelos pequenos às vezes fogem do formato, por isso há salvaguardas:
- se o primeiro passo não tiver uma Action válida, o agente busca no documento;
- se acabarem os passos, ele responde com o que já observou.
"""
import ast
import operator
import re
from dataclasses import dataclass
from typing import Callable, Iterator

from . import config

AGENT_PROMPT = """You are a research assistant that answers questions about PDF documents.
You can use these tools:
{tools}

Use exactly this format:
Thought: what you need to do next
Action: tool_name[input]

After each Action you will receive an Observation. When you know the answer, write:
Thought: I know the answer
Final Answer: the answer, in the same language as the question, citing pages like (p. 12)

Question: {question}
{scratchpad}"""

ACTION = re.compile(r"Action:\s*([a-z_]+)\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL)
FINAL = re.compile(r"Final Answer:\s*(.*)", re.IGNORECASE | re.DOTALL)
THOUGHT = re.compile(r"Thought:\s*(.*?)(?:\n|$)", re.IGNORECASE)


@dataclass
class Tool:
    name: str
    description: str
    run: Callable[[str], str]


@dataclass
class Step:
    """Um evento do agente, mostrado na interface conforme acontece."""
    kind: str      # "thought" | "action" | "observation" | "final"
    content: str
    tool: str = ""
    sources: list | None = None


def parse_output(text: str) -> tuple[str, str | None, str | None, str | None]:
    """Lê a saída do LLM. Retorna (thought, tool, tool_input, final_answer).

    Qualquer coisa depois da primeira Action é descartada: o modelo às vezes
    "inventa" a Observation em vez de esperar a ferramenta.
    """
    thought = (m.group(1).strip() if (m := THOUGHT.search(text)) else "")
    action = ACTION.search(text)
    final = FINAL.search(text)
    if action and (not final or action.start() < final.start()):
        return thought, action.group(1).lower(), action.group(2).strip().strip("\"'"), None
    if final:
        return thought, None, None, final.group(1).strip()
    return thought, None, None, None


# ---------- calculadora segura (sem eval) ----------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_calc(expression: str) -> str:
    """Avalia +, -, *, /, **, % e parênteses. Aceita vírgula decimal."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("expressão não permitida")

    try:
        result = ev(ast.parse(expression.replace(",", ".").replace("^", "**"), mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError) as e:
        return f"Erro: {e}"
    return f"{result:.6g}"


class Agent:
    def __init__(self, rag, llm: Callable[[str, int], str] | None = None,
                 max_steps: int = config.AGENT_MAX_STEPS, k: int = config.TOP_K):
        """llm(prompt, max_new_tokens) -> texto. Padrão: o gerador do RAG."""
        self.rag = rag
        self.llm = llm or (lambda p, n: rag.generator.generate(p, max_new_tokens=n))
        self.max_steps = max_steps
        self.k = k
        self.tools = {t.name: t for t in self._make_tools()}

    # ---------- ferramentas ----------

    def _make_tools(self) -> list[Tool]:
        return [
            Tool("search", "search the active document; input is a short search query",
                 self._search),
            Tool("search_all", "search ALL indexed documents; use to compare documents",
                 self._search_all),
            Tool("read_page", "read the text of one page of the active document; input is "
                 "the page number", self._read_page),
            Tool("calculator", "evaluate arithmetic, e.g. 2000*0.15", safe_calc),
        ]

    def _format(self, chunks: list[dict], with_doc: bool = False) -> str:
        self._last_sources = chunks
        lines = []
        for c in chunks:
            doc = f"{c.get('document', '')}, " if with_doc else ""
            lines.append(f"({doc}p. {c['page_number']}) {c['sentence_chunk'][:700]}")
        return "\n".join(lines) or "No results."

    def _search(self, query: str) -> str:
        return self._format(self.rag.search(query, self.k))

    def _search_all(self, query: str) -> str:
        return self._format(self.rag.search_all(query, self.k), with_doc=True)

    def _read_page(self, page: str) -> str:
        try:
            n = int(re.sub(r"\D", "", page))
        except ValueError:
            return "Erro: informe o número da página."
        chunks = [c for c in self.rag.chunks if c["page_number"] == n]
        return self._format(chunks[:3]) if chunks else f"Page {n} not found in the index."

    # ---------- laço ReAct ----------

    def run(self, question: str) -> Iterator[Step]:
        tools = "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())
        scratchpad = ""
        observations = []
        for step in range(self.max_steps):
            prompt = AGENT_PROMPT.format(tools=tools, question=question, scratchpad=scratchpad)
            output = self.llm(prompt, config.AGENT_STEP_TOKENS)
            thought, tool, tool_input, final = parse_output(output)

            if final is not None and observations:
                yield Step("thought", thought or "Já tenho informação suficiente.")
                yield Step("final", final)
                return
            if tool not in self.tools:
                # fora do formato (ou respondeu sem consultar nada): busca no documento
                thought = thought or "Vou buscar no documento."
                tool, tool_input = "search", (tool_input or question)
            if not tool_input:
                tool_input = question

            yield Step("thought", thought)
            yield Step("action", tool_input, tool=tool)
            self._last_sources = []
            observation = self.tools[tool].run(tool_input)
            observations.append(observation)
            yield Step("observation", observation, tool=tool, sources=self._last_sources)
            scratchpad += (f"Thought: {thought}\nAction: {tool}[{tool_input}]\n"
                           f"Observation: {observation}\n")

        # acabaram os passos: responde com o que já foi observado
        from .generation import BASE_PROMPT
        answer = self.llm(BASE_PROMPT.format(context="\n".join(observations), query=question),
                          config.MAX_NEW_TOKENS)
        yield Step("final", answer)
