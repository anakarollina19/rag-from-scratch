"""Configurações centrais do pipeline."""
import torch

# PDF de exemplo, usado quando `build` é chamado sem arquivo
BOOK_URL = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
BOOK_PATH = "data/human-nutrition-text.pdf"
BOOK_PAGE_OFFSET = 42  # a página 1 do livro é a página 43 do PDF

SENTENCES_PER_CHUNK = 10
MIN_TOKEN_LENGTH = 30

# Modelos de embedding suportados. Os da família E5 foram treinados com
# prefixos que dizem se o texto é uma pergunta ou um trecho de documento.
EMBEDDING_MODELS = {
    "all-mpnet-base-v2": {"query_prefix": "", "passage_prefix": ""},  # só inglês
    "intfloat/multilingual-e5-base": {"query_prefix": "query: ", "passage_prefix": "passage: "},
}
# Escolhido pela avaliação (eval/run_eval.py): melhor em inglês e em português
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
INDEX_DIR = "data/index"  # um arquivo .pt por PDF indexado

# Re-ranking: a busca vetorial traz RERANK_CANDIDATES trechos e um cross-encoder,
# que lê pergunta + trecho juntos, escolhe os TOP_K melhores
RERANK = True  # +9 pontos de hit@3 em português por ~1,4 s a mais por pergunta (CPU)
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # multilíngue
RERANK_CANDIDATES = 20

# Gemma exige aceitar a licença no Hugging Face e fazer `hf auth login`
LLM_MODEL = "google/gemma-2b-it"
MAX_NEW_TOKENS = 512
TOP_K = 3  # mais trechos = mais contexto, mas o Gemma 2B se perde com prompts longos

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
