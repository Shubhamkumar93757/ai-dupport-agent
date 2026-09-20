"""
Step 1: Parse pre-reconstructed conversations into structured threads.

The TNE-AI dataset already has full conversations in a text format like:
  Customer: message1
  Support: reply1
  Customer: message2
  ...

This script parses them into a structured JSON format for downstream use.

Output: data/threads.json
"""

import os
import re
import json
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INPUT_FILE = os.path.join(DATA_DIR, "brand_conversations.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "threads.json")


def parse_conversation(text: str) -> list[dict]:
    """
    Parse a conversation string into a list of messages.
    
    Input format:
      "Customer: msg1\nSupport: msg2\nCustomer: msg3\n..."
    
    Returns:
      [{"role": "customer"|"brand", "text": "..."}, ...]
    """
    messages = []
    # Split by the role prefixes while keeping the prefix
    # Handles multi-line messages within a single role
    parts = re.split(r'\n(?=Customer:|Support:)', text.strip())
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        if part.startswith("Customer:"):
            role = "customer"
            msg_text = part[len("Customer:"):].strip()
        elif part.startswith("Support:"):
            role = "brand"
            msg_text = part[len("Support:"):].strip()
        else:
            # Continuation of previous message or malformed
            if messages:
                messages[-1]["text"] += " " + part
            continue
        
        if msg_text:
            messages.append({"role": role, "text": msg_text})
    
    return messages


def main():
    print("[1/3] Loading brand conversations...")
    df = pd.read_csv(INPUT_FILE)
    print(f"       {len(df)} conversations loaded")

    # ── Parse conversations ────────────────────────────────────────────
    print("[2/3] Parsing conversations into structured threads...")
    threads = []
    skipped = 0

    for _, row in df.iterrows():
        convo_id = str(row["conversation_id"])
        convo_text = str(row.get("conversation", ""))
        
        if not convo_text or convo_text == "nan":
            skipped += 1
            continue
        
        messages = parse_conversation(convo_text)
        
        # Need at least one customer and one brand message
        has_customer = any(m["role"] == "customer" for m in messages)
        has_brand = any(m["role"] == "brand" for m in messages)
        
        if has_customer and has_brand and len(messages) >= 2:
            threads.append({
                "thread_id": convo_id,
                "messages": messages,
            })
        else:
            skipped += 1

    print(f"       Parsed {len(threads)} threads (skipped {skipped})")
    print(f"       Average thread length: {sum(len(t['messages']) for t in threads) / max(len(threads), 1):.1f} messages")

    # ── Save ───────────────────────────────────────────────────────────
    print(f"[3/3] Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(threads, f, indent=2)
    print(f"       Done! {len(threads)} threads saved.\n")

    # Show examples
    print("── Sample Threads ──────────────────────────────────────")
    for i, thread in enumerate(threads[:3]):
        print(f"\n  Thread [{i+1}] ({len(thread['messages'])} messages):")
        for msg in thread["messages"][:4]:
            prefix = "  👤" if msg["role"] == "customer" else "  🍎"
            print(f"    {prefix} {msg['text'][:120]}...")


if __name__ == "__main__":
    main()
