"""Configurações centrais do pipeline."""
import torch

PDF_URL = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
PDF_PATH = "data/human-nutrition-text.pdf"
PAGE_OFFSET = 41  # o conteúdo do livro começa na página 42 do PDF

SENTENCES_PER_CHUNK = 10
MIN_TOKEN_LENGTH = 30

EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBEDDINGS_PATH = "data/embeddings.pt"

# Gemma exige aceitar a licença no Hugging Face e fazer `huggingface-cli login`
LLM_MODEL = "google/gemma-2b-it"
MAX_NEW_TOKENS = 512
TOP_K = 3  # mais trechos = mais contexto, mas o Gemma 2B se perde com prompts longos

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
