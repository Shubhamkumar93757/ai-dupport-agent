"""
pipeline.py — Main orchestrator for the AI customer-support agent.

handle_message(text) → {intent, confidence, retrieved_cases, draft_reply, decision, reason}

Runs the full online pipeline: classify → retrieve → generate → decide.
Logs each intermediate step for easy debugging.
"""

import json
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.classifier import classify_intent
from agent.retriever import retrieve_similar_cases
from agent.reply_generator import generate_reply
from agent.escalation import decide_escalation


def handle_message(text: str) -> dict:
    """
    Process a single incoming customer message through the full pipeline.

    Returns:
        {
            "intent_breakdown": list of {"intent": str, "confidence": float},
            "intent": str,           # primary intent (top-1)
            "confidence": float,     # primary confidence
            "retrieved_cases": list of {customer_problem, brand_resolution, similarity},
            "draft_reply": str,
            "decision": "auto" | "escalate",
            "reason": str,
        }
    """
    print("\n" + "=" * 70)
    print(f'📩 INCOMING MESSAGE: "{text}"')
    print("=" * 70)

    # ── Step 1: Classify intent ────────────────────────────────────────
    print("\n🔍 Step 1: Classifying intent (top-3)...")
    classification = classify_intent(text)
    intent = classification[0]["intent"]
    confidence = classification[0]["confidence"]
    print(f"   Primary Intent: {intent} ({confidence:.2f})")
    for i, c in enumerate(classification[1:], 2):
        print(f"   #{i}: {c['intent']} ({c['confidence']:.2f})")

    # ── Step 2: Retrieve similar cases ─────────────────────────────────
    print("\n📚 Step 2: Retrieving similar historical cases...")
    retrieved_cases = retrieve_similar_cases(text, intent)
    print(f"   Found {len(retrieved_cases)} cases:")
    for i, case in enumerate(retrieved_cases, 1):
        print(f"   [{i}] (similarity={case['similarity']:.3f}, intent={case['intent']})")
        print(f"       Customer: {case['customer_problem'][:100]}...")
        print(f"       Resolution: {case['brand_resolution'][:100]}...")

    # ── Step 3: Generate grounded reply ────────────────────────────────
    print("\n✍️  Step 3: Generating draft reply...")
    draft_reply = generate_reply(text, intent, retrieved_cases)
    print(f'   Draft: "{draft_reply}"')

    # ── Step 4: Decide auto-handle vs. escalate ────────────────────────
    print("\n⚖️  Step 4: Deciding auto-handle vs. escalate...")
    escalation = decide_escalation(text, intent, confidence, retrieved_cases, draft_reply)
    decision = escalation["decision"]
    reason = escalation["reason"]
    emoji = "✅ AUTO" if decision == "auto" else "🚨 ESCALATE"
    print(f"   Decision: {emoji}")
    print(f"   Reason:   {reason}")

    # ── Structured output ──────────────────────────────────────────────
    result = {
        "intent_breakdown": classification,
        "intent": intent,
        "confidence": confidence,
        "retrieved_cases": retrieved_cases,
        "draft_reply": draft_reply,
        "decision": decision,
        "reason": reason,
    }

    print("\n" + "─" * 70)
    print("📦 STRUCTURED OUTPUT:")
    # Print a clean version without the full retrieved cases text
    clean_output = {**result, "retrieved_cases": f"[{len(retrieved_cases)} cases]"}
    print(json.dumps(clean_output, indent=2))
    print("─" * 70 + "\n")

    return result


# ── Test with example messages ─────────────────────────────────────────
if __name__ == "__main__":
    test_messages = [
        # Battery / power issue
        "My iPhone battery dies in 2 hours after the iOS update. This is ridiculous!",

        # Software update complaint
        "Ever since I updated to iOS 17, my phone is incredibly slow and apps keep crashing.",

        # Connectivity issue
        "My WiFi keeps disconnecting on my MacBook after the latest update. Very frustrating.",

        # Account / security
        "I can't log into my Apple ID. It says my account has been locked for security reasons.",

        # Safety / escalation trigger (should auto-escalate)
        "Your product is so bad I want to sue you. I'm contacting my lawyer about this.",
    ]

    for msg in test_messages:
        try:
            result = handle_message(msg)
        except Exception as e:
            print(f"\n❌ ERROR processing message: {e}")
            import traceback
            traceback.print_exc()
