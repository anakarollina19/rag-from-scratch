# 🔎 RAG from Scratch

Pipeline de **Retrieval-Augmented Generation (RAG)** construído do zero e rodando 100% local: sem LangChain, sem banco vetorial, sem API paga.

Funciona com **qualquer PDF**: o sistema lê o documento, cria embeddings dos trechos e responde perguntas usando o **Gemma 2B** com base no conteúdo encontrado, mostrando as páginas de onde tirou a resposta. Pode ser usado pela **interface web (Gradio)** ou pela linha de comando. Como exemplo, o projeto vem configurado com um livro de nutrição de licença aberta (~1200 páginas).

## Exemplo

```text
$ python main.py ask "How does saliva help with digestion?" --sources

Resposta:
Saliva helps with digestion by breaking down carbohydrates into smaller molecules
that can be absorbed by the body. It contains the enzyme, salivary amylase, which
breaks down starches into smaller chains of glucose. [...]

Fontes:
  [p. 240] score=0.638 Digestion and Absorption of Carbohydrates [...]
  [p. 72]  score=0.610 From the Mouth to the Stomach. There are four steps in the digestion process [...]
  [p. 240] score=0.588 But mechanical breakdown is ongoing [...]
```

## Como funciona

```
PDF ──► texto por página ──► sentenças (spaCy) ──► chunks de 10 sentenças
                                                         │
                                                         ▼
Pergunta ──► embedding ──► similaridade (dot product) ◄── embeddings (all-mpnet-base-v2)
                                  │
                                  ▼
                        top-3 trechos + pergunta ──► prompt ──► Gemma 2B ──► resposta
```

| Etapa | Arquivo | O que faz |
|---|---|---|
| 1. Ingestão | `rag/ingest.py` | Baixa o PDF, extrai o texto (PyMuPDF), quebra em sentenças e chunks e descarta trechos curtos |
| 2. Embeddings | `rag/embeddings.py` | Gera vetores de 768 dimensões com `sentence-transformers` e salva em `data/embeddings.pt` |
| 3. Busca | `rag/retrieval.py` | Calcula o produto escalar entre a pergunta e os chunks e retorna os top-k |
| 4. Geração | `rag/generation.py` | Monta o prompt com o contexto, gera a resposta com o Gemma e limpa frases de abertura |
| Tudo junto | `rag/pipeline.py` | Classe `RAG` que une as etapas; usada pela CLI (`main.py`) e pela interface web (`app.py`) |

Os principais parâmetros (modelos, tamanho dos chunks, `TOP_K`) ficam em `rag/config.py`. Cada PDF indexado vira um arquivo em `data/index/`, então dá para alternar entre documentos sem reprocessar.

## Instalação

Requer Python 3.10+.

```bash
git clone https://github.com/<seu-usuario>/rag-from-scratch.git
cd rag-from-scratch
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

O Gemma é um modelo com acesso controlado. Antes de usar:
1. aceite a licença em [huggingface.co/google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it);
2. faça login com a mesma conta:

```bash
hf auth login
```

## Uso

### Interface web

```bash
python app.py
```

Abra **http://127.0.0.1:7860** no navegador e:
1. envie um PDF em **"Indexar um novo PDF"** (ou escolha um já indexado);
2. digite a pergunta e clique em **Perguntar**.

A resposta aparece com os trechos e as páginas usadas. Desmarque **"Gerar resposta com o Gemma"** para ver só o resultado da busca, que é instantâneo.

### Linha de comando

```bash
python main.py build                                       # indexa o livro de exemplo (baixa o PDF)
python main.py build meu-arquivo.pdf                       # indexa qualquer PDF
python main.py build livro.pdf --page-offset 12            # desconta 12 páginas iniciais (capa, sumário...)
python main.py search "macronutrients functions"           # só a busca semântica
python main.py ask "How does saliva help with digestion?" --sources
python main.py chat                                        # modo interativo
python -m unittest                                         # testes (não baixam modelos)
```

`search`, `ask` e `chat` usam o PDF indexado mais recentemente. Para escolher outro, use `--index data/index/<nome>.pt`.

**Tempos medidos em CPU (sem GPU):**

| Comando | Tempo |
|---|---|
| `build` (1680 chunks) | ~5 min |
| `ask` (com o modelo já baixado) | ~1 min |

Na primeira execução, o `ask` também baixa o Gemma (~5 GB). Com GPU, os dois comandos ficam muito mais rápidos.

## O que aprendi

**1. O prompt "melhor" para um modelo grande pode quebrar um modelo pequeno.**
O notebook original usava um prompt longo, com instruções e três exemplos de resposta (*few-shot*). Com ele, o Gemma 2B respondia *"the passage does not provide information about how saliva helps with digestion"*, mesmo com os trechos certos no prompt. Para descobrir a causa, separei as etapas:
- a **busca** estava certa: os scores eram altos e os trechos falavam da amilase salivar;
- o **contexto chegava ao modelo**: imprimi o prompt final (~1300 tokens);
- com um **prompt curto e o mesmo contexto**, o modelo respondia bem.

O problema estava no tamanho e na complexidade do prompt, não na busca. Troquei por um prompt curto e direto.

**2. Mais contexto nem sempre é melhor.**
Com 5 trechos (`TOP_K=5`), a pergunta sobre saliva ainda falhava; com 3, funcionou. Os trechos extras traziam ruído (cabeçalhos, assuntos vizinhos), e um modelo pequeno se perde com isso. Em RAG há sempre esse equilíbrio entre *recall* (trazer tudo o que importa) e precisão (não trazer o que atrapalha).

**3. Sem sorteio, as respostas ficam previsíveis.**
Com `temperature=0.7`, o modelo sorteia as palavras e pode responder de jeitos diferentes a cada vez. Com *greedy decoding* (`do_sample=False`), ele sempre escolhe a palavra mais provável, e as respostas ficam estáveis e reproduzíveis.

**4. Integrações escondem detalhes de tipo.**
Ao enviar um PDF pela interface, o índice era salvo, mas não podia ser lido de volta. O Gradio entrega o caminho do arquivo como uma *subclasse* de `str`, e esse objeto ia parar dentro do arquivo `.pt`. Por segurança, o `torch.load` só aceita tipos básicos (`weights_only=True`) e recusava a leitura. A correção foi uma linha (`str(pdf_path)`); o difícil foi achar a causa, lendo o *traceback* até o fim.

**5. Bugs do notebook original corrigidos:**
- regex `\.(A-Z)` → `\.([A-Z])`;
- sentenças unidas sem espaço;
- `import numpy` faltando;
- `cuda` fixo no código (quebrava sem GPU);
- embeddings salvos em CSV (viravam texto); agora vão para um arquivo `.pt`.

## Limitações

- **Idioma:** só funciona bem com perguntas em inglês, porque o livro e o modelo de embeddings são em inglês.
- **Um documento por vez:** cada pergunta é feita sobre um único PDF, não sobre vários juntos.
- **PDFs escaneados:** PDFs que são só imagens (sem texto selecionável) não funcionam, porque não há OCR.
- **Respostas imperfeitas:** o Gemma 2B é pequeno e às vezes ainda diz que o contexto não tem a informação.

## Próximos passos

- [x] Aceitar qualquer PDF
- [x] Interface web com Gradio
- [ ] GIF da interface no topo do README
- [ ] Re-ranking dos resultados com um cross-encoder
- [ ] Embeddings multilíngues, para perguntas em português
- [ ] Conjunto de avaliação, para medir a qualidade das respostas

## Créditos

- Baseado no notebook [RAG From Scratch](https://www.kaggle.com/code/anshulgupta1502/rag-from-scratch), de Anshul Gupta (Apache 2.0), reorganizado em módulos e com as correções descritas acima.
- Dados: [Human Nutrition: 2020 Edition](https://pressbooks.oer.hawaii.edu/humannutrition2/), University of Hawai'i at Mānoa (CC BY 4.0).

## Licença

[Apache 2.0](LICENSE)
