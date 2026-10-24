"""Etapa 4: montar o prompt com o contexto e gerar a resposta com o LLM."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config

# Prompt curto de propósito: o Gemma 2B se confunde com instruções longas e
# exemplos (few-shot) e passa a responder que "o contexto não tem a informação".
BASE_PROMPT = """Context:
{context}

Using the context above, answer the question in a clear and explanatory way.
Question: {query}"""


def prompt_formatter(query: str, context_items: list[dict]) -> str:
    context = "- " + "\n- ".join(item["sentence_chunk"] for item in context_items)
    return BASE_PROMPT.format(context=context, query=query)


class Generator:
    def __init__(self, model_id: str = config.LLM_MODEL):
        dtype = torch.float16 if config.DEVICE == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype).to(config.DEVICE)

    def generate(self, prompt: str, max_new_tokens: int = config.MAX_NEW_TOKENS) -> str:
        # Gemma instruct espera o formato de chat
        chat = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(chat, tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(config.DEVICE)
        with torch.no_grad():
            # greedy (sem sorteio): respostas mais estáveis em modelos pequenos
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                         do_sample=False)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
