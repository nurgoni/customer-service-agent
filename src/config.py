# src/config.py

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = "qwen3.5:9b"            # model utama untuk agent
OPENAI_MODEL_FAST = "qwen3.5:9b"      # model untuk intent classifier (bisa lebih murah)
DB_PATH = "data/cs_agent.db"
EMBEDDING_MODEL = "qwen3-embedding:8b"
EMBEDDING_DIM = 384