"""
Step 2: Clean threads and create (customer_problem → brand_resolution) pairs.

- Strips @mentions, URLs, boilerplate
- Collapses each thread into one pair:
    customer_problem = customer's opening message + follow-ups
    brand_resolution = brand's substantive resolving reply
- Filters out threads with no substantive resolution

Output: data/paired_cases.csv
"""

import os
import re
import json
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INPUT_FILE = os.path.join(DATA_DIR, "threads.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "paired_cases.csv")

# Boilerplate patterns that indicate a non-substantive reply
BOILERPLATE_PATTERNS = [
    r"please\s+dm\s+us",
    r"send\s+us\s+a\s+dm",
    r"dm\s+(me|us)\s+(your|the)",
    r"join\s+us\s+in\s+dm",
    r"we('re|'d)\s+(love|like|happy)\s+to\s+help.*dm",
    r"https://t\.co/\S+",  # just a link with no real content
]


def clean_text(text: str) -> str:
    """Remove @mentions, URLs, and extra whitespace."""
    text = re.sub(r"@\w+", "", text)           # @mentions
    text = re.sub(r"https?://\S+", "", text)    # URLs
    text = re.sub(r"#\w+", "", text)            # hashtags
    text = re.sub(r"\^[A-Z]{2,3}\b", "", text)  # agent initials like ^RR, ^MM
    text = re.sub(r"&amp;", "&", text)          # HTML entities
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"\s+", " ", text).strip()    # collapse whitespace
    return text


def is_boilerplate(text: str) -> bool:
    """Check if a reply is just boilerplate (DM us, link only, etc.)."""
    cleaned = clean_text(text).lower()
    if len(cleaned) < 15:
        return True
    for pattern in BOILERPLATE_PATTERNS:
        if re.search(pattern, cleaned):
            # Check if there's substantial content beyond the boilerplate
            stripped = re.sub(pattern, "", cleaned).strip()
            if len(stripped) < 20:
                return True
    return False


def main():
    print("[1/3] Loading threads...")
    with open(INPUT_FILE, "r") as f:
        threads = json.load(f)
    print(f"       {len(threads)} threads loaded")

    # ── Build pairs ────────────────────────────────────────────────────
    print("[2/3] Cleaning and pairing...")
    pairs = []

    for thread in threads:
        msgs = thread["messages"]

        # Separate customer and brand messages
        customer_msgs = [m for m in msgs if m["role"] == "customer"]
        brand_msgs = [m for m in msgs if m["role"] == "brand"]

        if not customer_msgs or not brand_msgs:
            continue

        # Customer problem: concatenate all customer messages
        customer_texts = [clean_text(m["text"]) for m in customer_msgs]
        customer_texts = [t for t in customer_texts if t]  # remove empty
        if not customer_texts:
            continue
        customer_problem = " | ".join(customer_texts)

        # Brand resolution: find the most substantive brand reply
        # Prefer the last non-boilerplate reply (more likely to be resolving)
        brand_resolution = None
        for m in reversed(brand_msgs):
            text = clean_text(m["text"])
            if text and not is_boilerplate(m["text"]):
                brand_resolution = text
                break

        # Fallback: if all are boilerplate, take the longest one
        if brand_resolution is None:
            brand_texts = [clean_text(m["text"]) for m in brand_msgs if clean_text(m["text"])]
            if brand_texts:
                brand_resolution = max(brand_texts, key=len)

        if not brand_resolution or len(brand_resolution) < 10:
            continue

        if len(customer_problem) < 10:
            continue

        pairs.append({
            "thread_id": thread["thread_id"],
            "customer_problem": customer_problem,
            "brand_resolution": brand_resolution,
        })

    print(f"       Created {len(pairs)} clean pairs (from {len(threads)} threads)")
    print(f"       Dropped {len(threads) - len(pairs)} threads (no substantive resolution)")

    # ── Save ───────────────────────────────────────────────────────────
    print(f"[3/3] Saving to {OUTPUT_FILE}...")
    df = pd.DataFrame(pairs)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"       Done! {len(pairs)} pairs saved.\n")

    # Show a few examples
    print("── Sample Pairs ──────────────────────────────────────")
    for i, pair in enumerate(pairs[:3]):
        print(f"\n  [{i+1}] Customer: {pair['customer_problem'][:150]}...")
        print(f"      Resolution: {pair['brand_resolution'][:150]}...")


if __name__ == "__main__":
    main()
