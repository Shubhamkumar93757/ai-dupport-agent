"""
Intent classifier — few-shot LLM approach.

classify_intent(message) → [{"intent": str, "confidence": float}, ...] (top 3)

Uses Gemini with the derived intent definitions + examples as context.
Designed to be swappable: could later add an embedding+kNN baseline.
"""

import json
from agent.config import LLM_MODEL, INTENTS_PATH, groq_client


def _load_intents() -> dict:
    """Load the intent definitions from the offline-generated JSON."""
    with open(INTENTS_PATH, "r") as f:
        return json.load(f)


def _build_classification_prompt(message: str, intents: dict) -> str:
    """Build a few-shot prompt for top-3 intent classification."""
    intent_block = ""
    for name, info in intents.items():
        examples = "\n".join(f'    - "{ex}"' for ex in info["examples"][:3])
        intent_block += (
            f"\n**{name}**: {info['definition']}\n"
            f"  Examples:\n{examples}\n"
        )

    return f"""You are a customer-support intent classifier for AppleSupport.

Given the customer message below, classify it into the TOP 3 most likely intents
from the list below. Return ONLY a valid JSON array of exactly 3 objects, each with
"intent" (string) and "confidence" (float 0-1). The confidences should reflect
relative likelihood and sum to approximately 1.0. Order from most to least likely.

Do NOT include any other text or markdown formatting — just the raw JSON array.

## Intents
{intent_block}

## Customer Message
\"{message}\"

## Output (JSON array only)
"""


def classify_intent(message: str) -> list[dict]:
    """
    Classify the intent of an incoming customer message.

    Returns: [{"intent": str, "confidence": float}, ...] (top 3, sorted by confidence)
    """
    intents = _load_intents()
    prompt = _build_classification_prompt(message, intents)

    if not groq_client:
        raise RuntimeError("Groq client is not initialized. Please set GROQ_API_KEY.")

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

    # Parse the JSON response
    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if the model wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]  # drop first ```json line
        raw = raw.rsplit("```", 1)[0]  # drop closing ```
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: if the model returns garbage, escalate with low confidence
        return [
            {"intent": "unknown", "confidence": 0.0},
            {"intent": "unknown", "confidence": 0.0},
            {"intent": "unknown", "confidence": 0.0},
        ]

    # Handle the model returning a single object instead of an array
    if isinstance(parsed, dict):
        parsed = [parsed]

    # Validate and normalize each entry
    results = []
    for item in parsed[:3]:
        intent_name = item.get("intent", "unknown")
        confidence = float(item.get("confidence", 0.0))

        # Cap confidence for unknown intents
        if intent_name not in intents:
            confidence = min(confidence, 0.3)

        results.append({
            "intent": intent_name,
            "confidence": confidence,
        })

    # Pad to exactly 3 entries if the model returned fewer
    while len(results) < 3:
        results.append({"intent": "unknown", "confidence": 0.0})

    # Sort by confidence descending
    results.sort(key=lambda x: x["confidence"], reverse=True)

    return results
