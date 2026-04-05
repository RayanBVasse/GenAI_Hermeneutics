import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

CANON_MODEL = "claude-sonnet-4-20250514"
CANON_TEMPERATURE = 0.3
CANON_MAX_TOKENS = 16384

RETRIEVAL_TOP_K = 5
