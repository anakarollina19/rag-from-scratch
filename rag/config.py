"""Configurações centrais do pipeline."""
import torch

# PDF de exemplo, usado quando `build` é chamado sem arquivo
BOOK_URL = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
BOOK_PATH = "data/human-nutrition-text.pdf"
BOOK_PAGE_OFFSET = 42  # a página 1 do livro é a página 43 do PDF

SENTENCES_PER_CHUNK = 10
MIN_TOKEN_LENGTH = 30

EMBEDDING_MODEL = "all-mpnet-base-v2"
INDEX_DIR = "data/index"  # um arquivo .pt por PDF indexado

# Gemma exige aceitar a licença no Hugging Face e fazer `hf auth login`
LLM_MODEL = "google/gemma-2b-it"
MAX_NEW_TOKENS = 512
TOP_K = 3  # mais trechos = mais contexto, mas o Gemma 2B se perde com prompts longos

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
