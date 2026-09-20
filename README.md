# AI Customer-Support Agent (AppleSupport) — Phase 1

An AI agent that handles incoming customer messages by:
1. **Classifying intent** (few-shot LLM, 8 data-derived intents)
2. **Retrieving similar past resolutions** (semantic search over real AppleSupport threads)
3. **Generating a grounded reply** (mirrors the brand's real tone, never invents facts)
4. **Deciding auto-handle vs. escalate** (deterministic rules + LLM judge)

## Setup (< 15 minutes)

### 1. Prerequisites
- Python 3.10+
- `uv` (fast package manager) — or regular `pip`

### 2. Create virtual environment & install dependencies
```bash
cd "ai support agent"

# Using uv (recommended — much faster)
~/.local/bin/uv venv .venv
source .venv/bin/activate
~/.local/bin/uv pip install -r requirements.txt

# Or using pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Add your Gemini API key
Edit `.env` and paste your key:
```
GEMINI_API_KEY=your_actual_key_here
```

### 4. Run the offline pipeline (builds the knowledge base)
Run these in order — each takes 1–3 minutes:
```bash
python offline/00_download_and_select_brand.py   # Download & filter to AppleSupport
python offline/01_reconstruct_threads.py          # Reconstruct conversation threads
python offline/02_clean_and_pair.py               # Clean & create problem→resolution pairs
python offline/03_derive_intents.py               # Cluster & derive intent taxonomy
python offline/04_build_vector_index.py           # Build ChromaDB vector index
```

### 5. Run the agent
```bash
python pipeline.py
```

## Usage

```python
from pipeline import handle_message

result = handle_message("My iPhone battery dies in 2 hours after the iOS update!")

# result = {
#     "intent": "battery_and_power",
#     "confidence": 0.92,
#     "retrieved_cases": [...],
#     "draft_reply": "We understand how important battery life is...",
#     "decision": "auto",
#     "reason": "Draft addresses the core concern with appropriate guidance."
# }
```

## Project Structure

```
├── .env                  # Gemini API key (you fill this in)
├── requirements.txt      # Python dependencies
├── pipeline.py           # handle_message() — main entry point
├── README.md             # This file
├── data/                 # Generated data (created by offline scripts)
│   ├── brand_tweets.csv
│   ├── threads.json
│   ├── paired_cases.csv
│   ├── intents.json
│   └── chroma_db/
├── offline/              # Run-once data processing scripts
│   ├── 00_download_and_select_brand.py
│   ├── 01_reconstruct_threads.py
│   ├── 02_clean_and_pair.py
│   ├── 03_derive_intents.py
│   └── 04_build_vector_index.py
└── agent/                # Online inference modules
    ├── config.py          # All tunables in one place
    ├── classifier.py      # Intent classification (few-shot LLM)
    ├── retriever.py       # Semantic retrieval (ChromaDB)
    ├── reply_generator.py # Grounded reply generation
    └── escalation.py      # Auto-handle vs. escalate decision
```

## Design Decisions

- **Brand**: AppleSupport — highest volume, English-only, bounded product surface.
- **LLM**: Gemini 2.0 Flash — fast, cheap, configurable in `agent/config.py`.
- **Embeddings**: `all-MiniLM-L6-v2` — 384-dim, runs locally, no API needed.
- **Vector store**: ChromaDB (persistent) — lightweight, no separate server.
- **Intent derivation**: KMeans clustering on embeddings + keyword-based auto-naming.
- **Escalation**: Two layers — deterministic safety/confidence rules first, then an LLM
  judge that actively looks for problems with the draft.
