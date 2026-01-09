#!/usr/bin/env python3
"""
Test script to check what max tokens parameter Gemini API actually accepts.
"""

import sys
import os

# Add the project root to PYTHONPATH
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

# Now we can import from backend
from backend.llm.llm_handler import LLMHandler

def test_gemini_token_parameters():
    """Test what max tokens parameters Gemini API actually accepts."""
    
    print("🔍 Testing Gemini Token Parameters...")
    
    # Initialize handler
    handler = LLMHandler()
    
    # Test models
    models_to_test = [
        "models/gemini-2.5-flash-lite",
        "models/gemini-3-flash-preview"
    ]
    
    for model in models_to_test:
        print(f"\n📋 Testing model: {model}")
        
        # Test with max_output_tokens parameter
        try:
            print("  🔹 Testing with max_output_tokens and a long prompt...")
            result = handler.create(
                provider="gemini",
                model=model,
                input="write me a funny 3 line joke with a punchline. Return exactly 3 lines. Each line must be at least 12 words. No bullets. The joke should be really finny and be related to mountains and hikimg but should only include mountan ranges in the apls and italy and andes. do not mention anything about people and cultures that might cause issues with diversity. the joke should nbe a wionter time jokw and not a summer time joke . use words that are eacy to understand and not too complicated . no po in the  answer and please make the joke that should really tickle the funny bone in every individual across all continents in the works",
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

        # Test with max_output_tokens parameter
        try:
            print("  🔹 Testing with max_output_tokens and a short prompt...")
            result = handler.create(
                provider="gemini",
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
        
        # Test with max_tokens parameter (replicate joke test)
        #try:
        #    print("  🔹 Testing with max_tokens...")
        #    result = handler.create(
        #        provider="gemini",
        #        model=model,
        #        input="write me a funny 3 line joke. Return exactly 3 lines. Each line must be at least 12 words. No bullets",
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
        #try:
        #    print("  🔹 Testing with max_completion_tokens ...")
        #    result = handler.create(
        #        provider="gemini",
        #        model=model,
        #        input="write me a scientific fact with a 3 line explanation. Return exactly 3 lines. Each line must be at least 12 words. No bullets", 
        #        max_completion_tokens=330,  # Test with max_completion_tokens
        #        stream=False
        #    )
        #    print(f"  ✅ max_completion_tokens: SUCCESS")
        #    print(f"  📝 Response: {result.output_text}")
        #    print(f"  📏 Response length: {len(result.output_text)} chars")
        #    print(f"  🔢 Token usage: {result.usage}")
        #    usage = result.usage or {}
        #    print(f"  📊 Completion tokens: {usage.get('completion_tokens') if isinstance(usage, dict) else getattr(usage, 'completion_tokens', None)}")
        #    print(f"  🏁 Finish reason: {getattr(result, 'finish_reason', 'N/A')}")
        #    print(f"  📄 Raw response: {result.__dict__}")
        #except Exception as e:
        #    print(f"  ❌ max_completion_tokens: ERROR - {e}")
        
        # Test with wrong parameter name
    #    try:
    #        print("  🔹 Testing with max_wrong_tokens (should fail)...")
    #        result = handler.create(
    #            provider="gemini",
    #            model=model,
    #            input="write me a funny 3 line joke. Return exactly 3 lines. Each line must be at least 12 words. No bullets",
    #            max_wrong_tokens=20,  # Test with wrong parameter
    #            stream=False
    #        )
    #        print(f"  ⚠️  max_wrong_tokens: UNEXPECTED SUCCESS")
    #        print(f"  📝 Response: {result.output_text}")
    #        print(f"  📏 Response length: {len(result.output_text)} chars")
    #        print(f"  🔢 Token usage: {result.usage}")
    #        usage = result.usage or {}
    #        print(f"  📊 Completion tokens: {usage.get('completion_tokens') if isinstance(usage, dict) else getattr(usage, 'completion_tokens', None)}")
    #        print(f"  🏁 Finish reason: {getattr(result, 'finish_reason', 'N/A')}")
    #        print(f"  📄 Raw response: {result.__dict__}")
    #        print(f"  🔍 Raw finish_reason: {getattr(getattr(result.raw, 'choices', [{}])[0], 'finish_reason', 'N/A') if result.raw and hasattr(result.raw, 'choices') and result.raw.choices else 'N/A'}")
    #        print(f"  🚨 This suggests Gemini is ignoring the parameter!")
    #    except Exception as e:
    #        print(f"  ✅ max_wrong_tokens: EXPECTED ERROR - {e}")
        
    #    print("-" * 60)

if __name__ == "__main__":
    test_gemini_token_parameters()
