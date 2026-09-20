"""
Semantic retriever over historical (customer_problem → brand_resolution) pairs.

retrieve_similar_cases(message, intent, top_k) → list of dicts
"""

import chromadb
from sentence_transformers import SentenceTransformer

from agent.config import CHROMA_DB_PATH, EMBEDDING_MODEL, TOP_K_RETRIEVAL


# Lazy-loaded singletons so we don't re-init on every call
_model = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = client.get_collection("support_cases")
    return _collection


def retrieve_similar_cases(
    message: str,
    intent: str = None,
    top_k: int = TOP_K_RETRIEVAL,
) -> list[dict]:
    """
    Embed the incoming message, search the ChromaDB vector index,
    and return the top-k most similar historical cases.

    If intent is provided, tries to filter by intent first.
    Falls back to unfiltered search if filtered results are too sparse.

    Returns: list of {customer_problem, brand_resolution, intent, similarity}
    """
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode(message).tolist()

    # Try intent-filtered search first
    results = None
    if intent and intent != "unknown":
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"intent": intent},
            )
            # If we got fewer than 2 results, fall back to unfiltered
            if not results["ids"][0] or len(results["ids"][0]) < 2:
                results = None
        except Exception:
            results = None

    # Unfiltered fallback
    if results is None:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

    # Format output
    cases = []
    if results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            # ChromaDB returns L2 distance; convert to a 0-1 similarity
            similarity = max(0.0, 1.0 - distance / 2.0)
            cases.append({
                "customer_problem": meta.get("customer_problem", ""),
                "brand_resolution": meta.get("brand_resolution", ""),
                "intent": meta.get("intent", ""),
                "similarity": round(similarity, 3),
            })

    return cases
