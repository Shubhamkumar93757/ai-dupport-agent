"""
Centralized configuration for the AI support agent.
All tunables live here — model names, paths, thresholds.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_PATH = str(DATA_DIR / "chroma_db")
INTENTS_PATH = DATA_DIR / "intents.json"
PAIRED_CASES_PATH = DATA_DIR / "paired_cases.csv"

# ── LLM ────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = "qwen/qwen3.8-27b"          # fast + cheap, swap as needed

try:
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception:
    groq_client = None
GROQ_EVAL_MODEL = "qwen/qwen3.8-27b"

# ── Embeddings ─────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"     # 384-dim, very fast

# ── Retrieval ──────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 5                       # how many cases to pull
SIMILARITY_THRESHOLD = 0.25               # below this → "no good match"

# ── Classification ─────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5                # below this → auto-escalate

# ── Brand ──────────────────────────────────────────────────────────────
BRAND = "AppleSupport"

# ── Escalation keywords ───────────────────────────────────────────────
SAFETY_KEYWORDS = [
    "kill", "suicide", "die", "hurt myself", "self-harm",
    "lawyer", "lawsuit", "legal action", "attorney",
    "threat", "bomb", "gun", "weapon",
    "fraud", "scam", "steal", "stolen identity",
]
