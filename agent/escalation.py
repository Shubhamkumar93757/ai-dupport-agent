"""
Escalation decision engine — two layers.

Layer 1: Deterministic rules (safety, low confidence, bad retrieval).
Layer 2: LLM judge that actively looks for reasons the draft could be wrong.

decide_escalation(...) → {"decision": "auto"|"escalate", "reason": str}
"""

import json
from agent.config import (
    LLM_MODEL,
    CONFIDENCE_THRESHOLD,
    SIMILARITY_THRESHOLD,
    SAFETY_KEYWORDS,
    groq_client
)


def _check_deterministic_rules(
    message: str,
    confidence: float,
    retrieved_cases: list[dict],
) -> dict | None:
    """
    Layer 1: hard rules that always escalate.
    Returns an escalation dict if triggered, None otherwise.
    """
    msg_lower = message.lower()

    # Safety / legal / threat language
    for keyword in SAFETY_KEYWORDS:
        if keyword in msg_lower:
            return {
                "decision": "escalate",
                "reason": f"Safety/legal trigger detected: '{keyword}' found in message.",
            }

    # Very low classification confidence
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "decision": "escalate",
            "reason": (
                f"Classification confidence too low ({confidence:.2f} < "
                f"{CONFIDENCE_THRESHOLD}). Cannot reliably identify intent."
            ),
        }

    # No good retrieval match
    if not retrieved_cases:
        return {
            "decision": "escalate",
            "reason": "No similar historical cases found in the knowledge base.",
        }

    best_similarity = max(c["similarity"] for c in retrieved_cases)
    if best_similarity < SIMILARITY_THRESHOLD:
        return {
            "decision": "escalate",
            "reason": (
                f"Best retrieval similarity ({best_similarity:.2f}) is below "
                f"threshold ({SIMILARITY_THRESHOLD}). Insufficient grounding."
            ),
        }

    return None  # rules didn't trigger


def _llm_judge(
    message: str,
    intent: str,
    retrieved_cases: list[dict],
    draft_reply: str,
) -> dict:
    """
    Layer 2: LLM judge that critically evaluates whether the draft
    should be auto-sent or escalated to a human.
    """
    cases_summary = "\n".join(
        f"  - Customer: {c['customer_problem'][:120]}... → Resolution: {c['brand_resolution'][:120]}..."
        for c in retrieved_cases[:3]
    )

    prompt = f"""You are a quality-assurance judge for an AI customer-support system (AppleSupport).

Your job is to ACTIVELY LOOK FOR REASONS the draft reply below could be WRONG,
MISLEADING, or INSUFFICIENT. Do NOT rubber-stamp it.

Escalate if ANY of these are true:
- The draft promises something AppleSupport can't verify (refund, replacement, etc.)
- The draft gives incorrect technical advice
- The customer's issue is complex / multi-faceted and needs a specialist
- The customer is clearly distressed and a human touch is needed
- The draft doesn't actually address the customer's core concern
- The intent classification seems wrong given the message content

## Customer Message
"{message}"

## Classified Intent
{intent}

## Retrieved Historical Cases
{cases_summary}

## Draft Reply
"{draft_reply}"

## Your Verdict
Return ONLY a valid JSON object: {{"decision": "auto" or "escalate", "reason": "one-sentence explanation"}}
No other text.
"""

    if not groq_client:
        return {
            "decision": "escalate",
            "reason": "Groq client not configured. Defaulting to escalate for safety.",
        }

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_completion_tokens=256,
        top_p=0.95,
        stream=False,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

    try:
        result = json.loads(raw)
        return {
            "decision": result.get("decision", "escalate"),
            "reason": result.get("reason", "LLM judge returned unclear verdict."),
        }
    except json.JSONDecodeError:
        return {
            "decision": "escalate",
            "reason": "LLM judge response was unparseable — escalating to be safe.",
        }


def decide_escalation(
    message: str,
    intent: str,
    confidence: float,
    retrieved_cases: list[dict],
    draft_reply: str,
) -> dict:
    """
    Two-layer escalation decision.

    Returns: {"decision": "auto"|"escalate", "reason": str}
    """
    # Layer 1: deterministic rules
    rule_result = _check_deterministic_rules(message, confidence, retrieved_cases)
    if rule_result is not None:
        return rule_result

    # Layer 2: LLM judge
    return _llm_judge(message, intent, retrieved_cases, draft_reply)
