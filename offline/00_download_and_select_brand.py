"""
Step 0: Download dataset from HuggingFace and filter to AppleSupport.

Uses the TNE-AI/customer-support-on-twitter-conversation dataset which has
pre-reconstructed conversation threads with columns:
  conversation_id, company, conversation, summary

Filters to AppleSupport and saves to data/brand_conversations.csv
"""

import os
import sys
import pandas as pd
from datasets import load_dataset

# ── Config ─────────────────────────────────────────────────────────────
BRAND = "AppleSupport"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "brand_conversations.csv")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1/4] Loading dataset from HuggingFace (TNE-AI)...")
    ds = load_dataset(
        "TNE-AI/customer-support-on-twitter-conversation",
        split="train",
    )
    df = ds.to_pandas()
    print(f"       Loaded {len(df)} conversations, columns: {list(df.columns)}")

    # ── Brand volume analysis ──────────────────────────────────────────
    print("\n[2/4] Top 20 brands by conversation volume:")
    brand_counts = df["company"].value_counts().head(20)
    for rank, (brand, count) in enumerate(brand_counts.items(), 1):
        marker = " ← SELECTED" if brand == BRAND else ""
        print(f"  {rank:2d}. {brand:<25s}  {count:>5d} conversations{marker}")

    # ── Filter to chosen brand ─────────────────────────────────────────
    print(f"\n[3/4] Filtering to {BRAND}...")
    brand_df = df[df["company"] == BRAND].copy()
    brand_df = brand_df.reset_index(drop=True)
    print(f"       Found {len(brand_df)} conversations")

    # Show a few examples
    print("\n       Sample conversations:")
    for i in range(min(3, len(brand_df))):
        convo = brand_df.iloc[i]["conversation"]
        print(f"\n       [{i+1}] {convo[:200]}...")

    # ── Save ───────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving to {OUTPUT_FILE}...")
    brand_df.to_csv(OUTPUT_FILE, index=False)
    print(f"       Done! {len(brand_df)} conversations saved.\n")

    # ── Brand selection rationale ──────────────────────────────────────
    print("=" * 60)
    print("BRAND SELECTION: AppleSupport")
    print("=" * 60)
    print(
        "AppleSupport was chosen because: (1) It has the highest conversation\n"
        "volume among all brands in this dataset, ensuring plenty of repeating\n"
        "patterns for retrieval. (2) Conversations are overwhelmingly in English.\n"
        "(3) The product surface is reasonably bounded — mostly iOS updates,\n"
        "battery issues, app crashes, and hardware problems for a single\n"
        "ecosystem (Apple)."
    )


if __name__ == "__main__":
    main()
