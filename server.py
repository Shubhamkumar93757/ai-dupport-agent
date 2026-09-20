"""
server.py — FastAPI server wrapping the AI support agent pipeline.

Endpoints:
  POST /api/analyze    — Run full pipeline on a customer message
  POST /api/evaluate   — Run batch evaluation on random samples
  GET  /api/intents    — Return intent definitions
  GET  /api/health     — Health check
"""

import os
import sys
import json
import time
import asyncio
import random
import pandas as pd
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.config import (
    INTENTS_PATH, PAIRED_CASES_PATH,
    groq_client, GROQ_EVAL_MODEL,
)
from agent.classifier import classify_intent
from agent.retriever import retrieve_similar_cases
from agent.reply_generator import generate_reply
from agent.escalation import decide_escalation

# ── App setup ──────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Support Agent API",
    version="1.0.0",
    description="RAG-based customer support agent for AppleSupport",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class EvaluateRequest(BaseModel):
    num_samples: int = Field(default=5, ge=1, le=20)


# ── Endpoints ──────────────────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze_message(req: AnalyzeRequest):
    """Run the full pipeline on a customer message."""
    try:
        # Run the blocking pipeline in a thread to keep the event loop free
        result = await asyncio.to_thread(_run_pipeline, req.message)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _run_pipeline(message: str) -> dict:
    """Synchronous pipeline execution."""
    # Step 1: Classify
    classification = classify_intent(message)
    intent = classification[0]["intent"]
    confidence = classification[0]["confidence"]

    # Step 2: Retrieve
    retrieved_cases = retrieve_similar_cases(message, intent)

    # Step 3: Generate reply
    draft_reply = generate_reply(message, intent, retrieved_cases)

    # Step 4: Escalation decision
    escalation = decide_escalation(
        message, intent, confidence, retrieved_cases, draft_reply
    )

    return {
        "intent_breakdown": classification,
        "intent": intent,
        "confidence": confidence,
        "retrieved_cases": retrieved_cases,
        "draft_reply": draft_reply,
        "escalation": {
            "decision": escalation["decision"],
            "reason": escalation["reason"],
        },
    }


@app.post("/api/evaluate")
async def run_evaluation(req: EvaluateRequest):
    """Run batch evaluation on random samples from the dataset."""
    if not groq_client:
        raise HTTPException(
            status_code=503,
            detail="Groq client not configured. Set GROQ_API_KEY in .env",
        )

    try:
        results = await asyncio.to_thread(_run_evaluation, req.num_samples)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _run_evaluation(num_samples: int) -> dict:
    """Synchronous batch evaluation."""
    df = pd.read_csv(PAIRED_CASES_PATH)
    sample_df = df.sample(n=min(num_samples, len(df))).reset_index(drop=True)

    results = []
    for idx, row in sample_df.iterrows():
        customer_msg = row["customer_problem"]
        actual_reply = row["brand_resolution"]

        try:
            # Run pipeline
            classification = classify_intent(customer_msg)
            intent = classification[0]["intent"]
            retrieved_cases = retrieve_similar_cases(customer_msg, intent)
            draft_reply = generate_reply(customer_msg, intent, retrieved_cases)

            # Evaluate with Groq
            eval_result = _evaluate_with_groq(draft_reply, actual_reply)

            results.append({
                "customer_message": customer_msg[:300],
                "actual_reply": actual_reply[:300],
                "draft_reply": draft_reply,
                "intent": intent,
                "score": eval_result.get("score", 0),
                "reasoning": eval_result.get("reasoning", ""),
                "error": None,
            })

        except Exception as e:
            results.append({
                "customer_message": customer_msg[:300],
                "actual_reply": actual_reply[:300],
                "draft_reply": "",
                "intent": "",
                "score": 0,
                "reasoning": "",
                "error": str(e),
            })

        # Rate limiting (Groq rate limits are higher, but can add a small sleep if needed)
        # time.sleep(1) 

    scores = [r["score"] for r in results if r["error"] is None and r["score"] > 0]
    avg_score = sum(scores) / len(scores) if scores else 0

    return {
        "results": results,
        "avg_score": round(avg_score, 1),
        "total": len(results),
        "successful": len(scores),
    }


def _evaluate_with_groq(draft_reply: str, ground_truth_reply: str) -> dict:
    """Uses Groq to compare the draft reply against the real brand resolution."""
    prompt = f"""You are an expert customer support evaluator.
Compare the drafted AI reply to the actual historical brand reply.

Draft AI Reply:
"{draft_reply}"

Actual Historical Brand Reply:
"{ground_truth_reply}"

Evaluate how well the drafted reply matches the intent, tone, and resolution of the actual reply.
Provide a JSON object with exactly two keys:
1. "score": An integer from 0 to 100 representing the semantic similarity and appropriateness.
2. "reasoning": A short sentence explaining why you gave this score.

Respond ONLY with valid JSON. Do not include markdown blocks like ```json.
"""
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_EVAL_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful JSON-only output assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_completion_tokens=2048,
            top_p=0.95,
            stream=False,
        )

        result_text = response.choices[0].message.content.strip()

        # Clean up if model still output markdown
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]

        return json.loads(result_text.strip())
    except Exception as e:
        return {"score": 0, "reasoning": f"Failed to evaluate: {e}"}


@app.get("/api/intents")
async def get_intents():
    """Return the intent definitions."""
    try:
        with open(INTENTS_PATH, "r") as f:
            intents = json.load(f)
        return {"intents": intents}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="intents.json not found")


@app.get("/api/health")
async def health_check():
    """Health check — verifies components are available."""
    status = {
        "status": "ok",
        "groq_configured": groq_client is not None,
        "intents_loaded": os.path.exists(INTENTS_PATH),
        "chroma_db_exists": os.path.exists(
            os.path.join(os.path.dirname(__file__), "data", "chroma_db")
        ),
    }
    return status


# ── Serve frontend static files ───────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend_assets")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
