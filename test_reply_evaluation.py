import os
import json
import time
import pandas as pd
from dotenv import load_dotenv

from agent.config import groq_client, GROQ_EVAL_MODEL
from agent.classifier import classify_intent
from agent.retriever import retrieve_similar_cases
from agent.reply_generator import generate_reply

# Load environment variables
load_dotenv()

if not groq_client:
    print("❌ ERROR: groq_client is not initialized. Ensure GROQ_API_KEY is in .env")
    exit(1)

def evaluate_reply_with_groq(draft_reply: str, ground_truth_reply: str) -> dict:
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
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_completion_tokens=2048,
            top_p=0.95,
            stream=False,
            stop=None,
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Clean up if model still output markdown
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
            
        return json.loads(result_text)
    except Exception as e:
        return {"score": 0, "reasoning": f"Failed to evaluate: {e}"}

def run_evaluation():
    print("Loading dataset...")
    df = pd.read_csv("data/paired_cases.csv")
    
    # Take 5 random samples
    sample_df = df.sample(n=5, random_state=42).reset_index(drop=True)
    
    print("Starting evaluation on 5 random cases...\n")
    
    for idx, row in sample_df.iterrows():
        print("=" * 70)
        print(f"CASE {idx + 1}")
        
        customer_msg = row['customer_problem']
        actual_reply = row['brand_resolution']
        
        print(f"Customer Message: {customer_msg[:150]}...")
        
        # Run pipeline steps
        print("\n⏳ Processing through pipeline...")
        try:
            classification = classify_intent(customer_msg)
            intent = classification[0]['intent']
            retrieved_cases = retrieve_similar_cases(customer_msg, intent)
            draft_reply = generate_reply(customer_msg, intent, retrieved_cases)
            
            print(f"Draft Reply:  {draft_reply}")
            print(f"Actual Reply: {actual_reply}")
            
            # Evaluate with Groq
            print("\n⚖️ Evaluating with Groq...")
            eval_result = evaluate_reply_with_groq(draft_reply, actual_reply)
            
            print(f"\n✅ Score: {eval_result.get('score')}/100")
            print(f"🧠 Reasoning: {eval_result.get('reasoning')}")
            
        except Exception as e:
            print(f"\n❌ Error processing case: {e}")
            
        print("=" * 70 + "\n")
        
        # Add a small delay for Groq rate limits if needed
        if idx < len(sample_df) - 1:
            time.sleep(1)

if __name__ == "__main__":
    run_evaluation()
