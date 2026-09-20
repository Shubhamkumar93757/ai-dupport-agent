"""
Grounded reply generator.

generate_reply(message, intent, retrieved_cases) → str

Prompts the LLM to draft a reply that mirrors AppleSupport's real tone
and resolution patterns, grounded in retrieved historical cases.
"""

import json
from agent.config import LLM_MODEL, groq_client


def _build_reply_prompt(
    message: str,
    intent: str,
    retrieved_cases: list[dict],
) -> str:
    """Build the grounded reply-generation prompt."""
    cases_block = ""
    for i, case in enumerate(retrieved_cases, 1):
        cases_block += (
            f"\n--- Case {i} (similarity: {case['similarity']}) ---\n"
            f"Customer: {case['customer_problem']}\n"
            f"AppleSupport reply: {case['brand_resolution']}\n"
        )

    return f"""You are a customer-support agent for AppleSupport on Twitter.

Draft a reply to the customer message below. Follow these rules strictly:

1. Mirror AppleSupport's real tone: helpful, empathetic, professional, concise.
2. Ground your reply in the historical resolution patterns shown below — adapt,
   don't copy verbatim.
3. Do NOT invent account-specific facts (order status, refund amounts, serial
   numbers, etc.) that you have no way of knowing. Instead, ask the customer
   for those details.
4. If the issue requires private information, suggest moving to DM.
5. Keep the reply under 280 characters if possible (it's Twitter).

## Classified Intent
{intent}

## Historical Resolutions (for reference)
{cases_block if cases_block else "No similar cases found."}

## Customer Message
"{message}"

## Your Draft Reply (plain text, no JSON)
"""


def generate_reply(
    message: str,
    intent: str,
    retrieved_cases: list[dict],
) -> str:
    """
    Generate a grounded draft reply using the LLM, informed by
    retrieved historical resolutions.
    """
    prompt = _build_reply_prompt(message, intent, retrieved_cases)

    if not groq_client:
        raise RuntimeError("Groq client is not initialized. Please set GROQ_API_KEY.")

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        max_completion_tokens=1024,
        top_p=0.95,
        stream=False,
    )
    return response.choices[0].message.content.strip()
