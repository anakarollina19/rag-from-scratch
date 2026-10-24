# 🔎 RAG from Scratch

Pipeline de **Retrieval-Augmented Generation (RAG)** construído do zero, rodando 100% local: sem LangChain, sem banco vetorial, sem API paga.

O sistema lê um livro de nutrição em PDF (~1200 páginas, licença aberta), cria embeddings dos trechos e responde perguntas usando o **Gemma 2B** com base no conteúdo encontrado.

## Como funciona

```
PDF ──► texto por página ──► sentenças (spaCy) ──► chunks de 10 sentenças
                                                         │
                                                         ▼
Pergunta ──► embedding ──► similaridade (dot product) ◄── embeddings (all-mpnet-base-v2)
                                  │
                                  ▼
                        top-k trechos + pergunta ──► prompt ──► Gemma 2B ──► resposta
```

| Etapa | Arquivo | O que faz |
|---|---|---|
| 1. Ingestão | `rag/ingest.py` | Baixa o PDF, extrai texto (PyMuPDF), quebra em sentenças e chunks, descarta trechos curtos |
| 2. Embeddings | `rag/embeddings.py` | Gera vetores de 768 dimensões com `sentence-transformers` e salva em `data/embeddings.pt` |
| 3. Busca | `rag/retrieval.py` | Produto escalar entre pergunta e chunks, retorna os top-k |
| 4. Geração | `rag/generation.py` | Monta o prompt com contexto + exemplos e gera a resposta com o Gemma |

## Instalação

```bash
git clone https://github.com/<seu-usuario>/rag-from-scratch.git
cd rag-from-scratch
python -m venv .venv
pip install -r requirements.txt
```

O Gemma é um modelo com acesso controlado: aceite a licença em [huggingface.co/google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it) e faça login:

```bash
huggingface-cli login
```

## Uso

```bash
python main.py build                                   # cria o índice (uma vez)
python main.py search "macronutrients functions"       # só a busca
python main.py ask "How does saliva help with digestion?" --sources
python main.py chat                                    # modo interativo
```

> GPU é recomendada para a geração. Em CPU funciona, mas é lento. Para trocar o modelo, edite `rag/config.py`.

## Possíveis melhorias

- Re-ranking dos resultados com um cross-encoder
- Índice FAISS para bases maiores
- Interface web com Gradio/Streamlit
- Suporte a outros PDFs e a perguntas em português

## Créditos

Inspirado no notebook [RAG From Scratch](https://www.kaggle.com/code/anshulgupta1502/rag-from-scratch) (Apache 2.0). Dados: [Human Nutrition: 2020 Edition](https://pressbooks.oer.hawaii.edu/humannutrition2/) (University of Hawai'i, CC BY 4.0).
