#!/usr/bin/env python3
"""
Test OpenAI token parameters and response structure.
Tests reasoning vs non-reasoning models and shows raw response details.
"""

import sys
import os

# Add the project root to PYTHONPATH
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)



from backend.llm.llm_client import generate

def test_openai_token_parameters():
    """Test OpenAI token parameters and response structure."""
    
    print("🔍 Testing OpenAI Token Parameters...")
    
    # Models to test - reasoning and non-reasoning
    models_to_test = [
        "gpt-4o",           # Non-reasoning model
        "o3-mini",         # Reasoning model
    ]
    
    handler = LLMHandler()
    
    for model in models_to_test:
        print(f"\n📋 Testing model: {model}")
        
        # Test with max_output_tokens parameter
        try:
            print("  🔹 Testing with max_output_tokens and a long prompt...")
            result = handler.create(
                provider="openai",
                model=model,
                input="write me a funny 3 line joke with a punchline. Return exactly 3 lines. Each line must be at least 12 words. No bullets. The joke should be really funny and be related to mountains and hiking but should only include mountain ranges in the Alps and Italy and Andes. Do not mention anything about people and cultures that might cause issues with diversity. The joke should be a winter time joke and not a summer time joke. Use words that are easy to understand and not too complicated. No typos in the answer and please make the joke that should really tickle the funny bone in every individual across all continents in the world",
                max_output_tokens=200,  # Test with max_output_tokens
                stream=False
            )
            print(f"  ✅ max_output_tokens: SUCCESS")
            print(f"  📝 Response: {result.output_text}")
            print(f"  📏 Response length: {len(result.output_text)} chars")
            print(f"  🔢 Token usage: {result.usage}")
            usage = result.usage or {}
            print(f"  📊 Completion tokens: {usage.get('completion_tokens') if isinstance(usage, dict) else getattr(usage, 'completion_tokens', None)}")
            print(f"  🏁 Finish reason: {getattr(result, 'finish_reason', 'N/A')}")
            print(f"  📄 Raw response: {result.__dict__}")
        except Exception as e:
            print(f"  ❌ max_output_tokens: ERROR - {e}")

        # Test with max_output_tokens parameter (short prompt)
        try:
            print("  🔹 Testing with max_output_tokens and a short prompt...")
            result = handler.create(
                provider="openai",
                model=model,
                input="write me a funny 3 line joke with a punchline.", 
                max_output_tokens=200,  # Test with max_output_tokens
                stream=False
            )
            print(f"  ✅ max_output_tokens: SUCCESS")
            print(f"  📝 Response: {result.output_text}")
            print(f"  📏 Response length: {len(result.output_text)} chars")
            print(f"  🔢 Token usage: {result.usage}")
            usage = result.usage or {}
            print(f"  📊 Completion tokens: {usage.get('completion_tokens') if isinstance(usage, dict) else getattr(usage, 'completion_tokens', None)}")
            print(f"  🏁 Finish reason: {getattr(result, 'finish_reason', 'N/A')}")
            print(f"  📄 Raw response: {result.__dict__}")
        except Exception as e:
            print(f"  ❌ max_output_tokens: ERROR - {e}")

        # Test with max_tokens parameter - OPENAI Responses API fails with "max_tokens" not a valid attribute. 
        #try:
        #    print("  🔹 Testing with max_tokens...")
        #    result = handler.create(
        #        provider="openai",
        #        model=model,
        #        input="write me a scientific fact with a 3 line explanation. Return exactly 3 lines. Each line must be at least 12 words. No bullets", 
        #        max_tokens=200,  # Test with max_tokens
        #        stream=False
        #    )
        #    print(f"  ✅ max_tokens: SUCCESS")
        #    print(f"  📝 Response: {result.output_text}")
        #    print(f"  📏 Response length: {len(result.output_text)} chars")
        #    print(f"  🔢 Token usage: {result.usage}")
        #    usage = result.usage or {}
        #    print(f"  📊 Completion tokens: {usage.get('completion_tokens') if isinstance(usage, dict) else getattr(usage, 'completion_tokens', None)}")
        #    print(f"  🏁 Finish reason: {getattr(result, 'finish_reason', 'N/A')}")
        #    print(f"  📄 Raw response: {result.__dict__}")
        #except Exception as e:
        #    print(f"  ❌ max_tokens: ERROR - {e}")

        # Test with max_completion_tokens parameter  
        try:
            print("  🔹 Testing with max_completion_tokens...")
            result = handler.create(
                provider="openai",
                model=model,
                input="write me a historical fact about ancient Rome with a 3 line explanation. Return exactly 3 lines. Each line must be at least 12 words. No bullets", 
                max_completion_tokens=330,  # Test with max_completion_tokens
                stream=False
            )
            print(f"  ✅ max_completion_tokens: SUCCESS")
            print(f"  📝 Response: {result.output_text}")
            print(f"  📏 Response length: {len(result.output_text)} chars")
            print(f"  🔢 Token usage: {result.usage}")
            usage = result.usage or {}
            print(f"  📊 Completion tokens: {usage.get('completion_tokens') if isinstance(usage, dict) else getattr(usage, 'completion_tokens', None)}")
            print(f"  🏁 Finish reason: {getattr(result, 'finish_reason', 'N/A')}")
            print(f"  📄 Raw response: {result.__dict__}")
        except Exception as e:
            print(f"  ❌ max_completion_tokens: ERROR - {e}")
        
        # Additional streaming test for reasoning model to exercise reasoning_effort mapping
        if model == "o3-mini":
            try:
                print("  🔹 Testing STREAMING with reasoning_effort='medium'...")
                stream = handler.create(
                    provider="openai",
                    model=model,
                    input="Give a step-by-step explanation of how to solve a quadratic equation.",
                    max_output_tokens=200,
                    reasoning_effort="medium",
                    stream=True,
                )

                full_text = ""
                for event in stream:
                    etype = getattr(event, "type", None)
                    if etype == "response.output_text.delta":
                        chunk = getattr(event, "delta", "") or ""
                        print(chunk, end="", flush=True)
                        full_text += chunk
                    elif etype == "response.output_text.done":
                        print("\n  ✅ Streaming completed")
                        break

                print(f"  📏 Streaming response length: {len(full_text)} chars")
            except Exception as e:
                print(f"  ❌ STREAMING with reasoning_effort: ERROR - {e}")
        
        print("-" * 60)

if __name__ == "__main__":
    test_openai_token_parameters()
