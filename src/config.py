import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

# --- LLM (OpenAI SDK; arahkan ke Ollama lewat OPENAI_BASE_URL) ---
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL") or None  # None = OpenAI asli
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MODEL_FAST = os.environ.get("OPENAI_MODEL_FAST", OPENAI_MODEL)

# Qwen3 adalah model "thinking": tanpa ini token habis untuk reasoning dan content bisa kosong.
# Default "none" hanya bila memakai base_url kustom (Ollama). Kosongkan untuk tidak mengirim parameter ini.
_default_effort = "none" if OPENAI_BASE_URL else ""
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", _default_effort).strip()
LLM_EXTRA = {"extra_body": {"reasoning_effort": LLM_REASONING_EFFORT}} if LLM_REASONING_EFFORT else {}

# --- Database ---
DB_PATH = BASE_DIR / "data" / "cs_agent.db"

# --- Embedding (endpoint /v1/embeddings yang kompatibel OpenAI, mis. Ollama) ---
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-m3:567m")
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL") or OPENAI_BASE_URL
# Dimensi vektor TIDAK di-hardcode: tabel vektor dibuat saat seed mengikuti dimensi model (bge-m3 = 1024).
