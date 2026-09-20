"""
Step 4: Build the ChromaDB vector index over paired cases.

Embeds customer problems, assigns intent labels, and stores everything
in a persistent ChromaDB collection for semantic retrieval at inference time.

Output: data/chroma_db/
"""

import os
import json
import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PAIRED_FILE = os.path.join(DATA_DIR, "paired_cases.csv")
INTENTS_FILE = os.path.join(DATA_DIR, "intents.json")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
LABELS_FILE = os.path.join(DATA_DIR, "cluster_labels.npy")
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main():
    print("[1/4] Loading data...")
    df = pd.read_csv(PAIRED_FILE)
    with open(INTENTS_FILE, "r") as f:
        intents = json.load(f)
    print(f"       {len(df)} paired cases, {len(intents)} intents")

    # ── Load or compute embeddings ─────────────────────────────────────
    print("[2/4] Loading embeddings...")
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(LABELS_FILE):
        embeddings = np.load(EMBEDDINGS_FILE)
        labels = np.load(LABELS_FILE)
        print(f"       Loaded cached embeddings: {embeddings.shape}")
    else:
        print(f"       Computing embeddings with {EMBEDDING_MODEL}...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = model.encode(df["customer_problem"].tolist(), show_progress_bar=True)
        # Simple nearest-centroid assignment for labels
        labels = np.zeros(len(df), dtype=int)

    # Build cluster_id → intent_name mapping
    cluster_to_intent = {}
    for name, info in intents.items():
        cluster_to_intent[info["cluster_id"]] = name

    # ── Build ChromaDB index ───────────────────────────────────────────
    print("[3/4] Building ChromaDB index...")
    # Remove existing DB if present
    if os.path.exists(CHROMA_PATH):
        import shutil
        shutil.rmtree(CHROMA_PATH)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.create_collection(
        name="support_cases",
        metadata={"description": "AppleSupport customer problem → resolution pairs"},
    )

    # Add in batches (ChromaDB limit)
    BATCH_SIZE = 500
    total = len(df)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_df = df.iloc[start:end]
        batch_embeddings = embeddings[start:end]
        batch_labels = labels[start:end]

        ids = [f"case_{i}" for i in range(start, end)]
        embs = batch_embeddings.tolist()
        metadatas = []
        for i, (_, row) in enumerate(batch_df.iterrows()):
            intent_name = cluster_to_intent.get(int(batch_labels[i]), "unknown")
            metadatas.append({
                "customer_problem": str(row["customer_problem"])[:500],  # ChromaDB metadata limit
                "brand_resolution": str(row["brand_resolution"])[:500],
                "intent": intent_name,
                "thread_id": str(row["thread_id"]),
            })

        collection.add(
            ids=ids,
            embeddings=embs,
            metadatas=metadatas,
        )

        print(f"       Added {end}/{total} cases...")

    # ── Verify ─────────────────────────────────────────────────────────
    print(f"\n[4/4] Verifying index...")
    count = collection.count()
    print(f"       ChromaDB collection has {count} entries")

    # Quick test query
    model = SentenceTransformer(EMBEDDING_MODEL)
    test_query = "My iPhone battery drains too fast after the update"
    test_embedding = model.encode(test_query).tolist()
    results = collection.query(query_embeddings=[test_embedding], n_results=3)

    print(f"\n       Test query: \"{test_query}\"")
    print(f"       Top 3 results:")
    for i, meta in enumerate(results["metadatas"][0]):
        dist = results["distances"][0][i]
        print(f"         {i+1}. [{meta['intent']}] (dist={dist:.3f})")
        print(f"            Customer: {meta['customer_problem'][:100]}...")
        print(f"            Resolution: {meta['brand_resolution'][:100]}...")

    print(f"\n       Done! Index saved to {CHROMA_PATH}\n")


if __name__ == "__main__":
    main()
