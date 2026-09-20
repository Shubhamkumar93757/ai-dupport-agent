import json
import time
from agent.classifier import classify_intent

def run_tests():
    test_cases = [
        {
            "description": "Battery and Power Issue",
            "message": "My iPhone battery dies within 2 hours after installing the latest update.",
            "expected_intent_keyword": "battery"
        },
        {
            "description": "Connectivity Issue",
            "message": "I can't connect to my home WiFi or use Bluetooth headphones on my iPad anymore.",
            "expected_intent_keyword": "connectivity"
        },
        {
            "description": "App and Store Issue",
            "message": "The App Store keeps telling me my payment method is invalid when I try to download apps.",
            "expected_intent_keyword": "app_and_store"
        },
        {
            "description": "Hardware and Display Issue",
            "message": "My screen is completely cracked and Face ID doesn't recognize my face anymore.",
            "expected_intent_keyword": "hardware"
        },
        {
            "description": "Account and Security Issue",
            "message": "I forgot my Apple ID password, and my account is locked. How can I recover it?",
            "expected_intent_keyword": "account"
        },
        {
            "description": "Software Update Issue",
            "message": "When will you release a patch for the weird text glitch on iOS 11?",
            "expected_intent_keyword": "software"
        },
        {
            "description": "Performance Issue",
            "message": "My phone is super slow, apps keep freezing and it crashes constantly.",
            "expected_intent_keyword": "performance"
        },
        {
            "description": "General Help Request",
            "message": "How do I add a new playlist to Apple Music? I just want some general guidance.",
            "expected_intent_keyword": "general_help"
        },
        {
            "description": "Unknown/Irrelevant Intent (Should have low confidence)",
            "message": "What is the capital of France and what is the weather there today?",
            "expected_intent_keyword": "unknown"
        }
    ]

    print("Running intent classifier tests (top-3 mode)...\n")
    for idx, test in enumerate(test_cases, 1):
        print(f"Test {idx}: {test['description']}")
        print(f"Message: {test['message']}")
        
        try:
            results = classify_intent(test['message'])
            # Primary intent is the first in the list
            primary = results[0]
            intent = primary.get('intent', 'unknown')
            confidence = primary.get('confidence', 0.0)
            
            print(f"Top-3 Intents:")
            for i, r in enumerate(results, 1):
                print(f"  #{i}: '{r['intent']}' ({r['confidence']:.2f})")
            
            if test['expected_intent_keyword'] in intent:
                print("Result: ✅ MATCHES EXPECTED CATEGORY")
            elif test['expected_intent_keyword'] == 'unknown' and confidence < 0.5:
                 print("Result: ✅ HANDLED AS LOW CONFIDENCE/UNKNOWN")
            else:
                print(f"Result: ⚠️ UNEXPECTED CATEGORY (Expected keyword: '{test['expected_intent_keyword']}')")
                
        except Exception as e:
            print(f"Result: ❌ ERROR: {e}")
            
        print("-" * 60)
        
        # Add a small delay for Groq rate limits if needed (usually fine without for small batches)
        if idx < len(test_cases):
            time.sleep(1)

if __name__ == "__main__":
    run_tests()
