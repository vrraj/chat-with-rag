#!/usr/bin/env python3
"""
Test script for OpenAI streaming functionality.
Tests both regular and reasoning models with streaming output.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

try:
    from backend.llm.llm_client import generate
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"📁 Script directory: {script_dir}")
    print(f"📁 Project root: {project_root}")
    print(f"📁 Project root exists: {project_root.exists()}")
    print(f"📁 Backend directory: {project_root / 'backend'}")
    print(f"📁 Backend exists: {(project_root / 'backend').exists()}")
    print(f"📁 LLM client exists: {(project_root / 'backend' / 'llm' / 'llm_client.py').exists()}")
    print(f"🐍 Python path: {sys.path[:3]}...")  # Show first 3 paths
    
    # Try alternative import
    try:
        print("\n🔄 Trying alternative import...")
        sys.path.insert(0, str(project_root / 'backend'))
        from llm.llm_client import generate
    except ImportError as e2:
        print(f"❌ Alternative import also failed: {e2}")
        sys.exit(1)


def test_openai_streaming():
    """Test OpenAI streaming with different models."""
    
    print("🚀 Testing OpenAI Streaming")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            "name": "OpenAI Fast Model (gpt-4o-mini)",
            "model_key": "openai:gpt-4o-mini",
            "prompt": "Write a short poem about artificial intelligence",
            "expected_tokens": 150
        },
        {
            "name": "OpenAI Direct Model (gpt-4o-mini)",
            "model_key": "openai:gpt-4o-mini", 
            "prompt": "Explain quantum computing in simple terms",
            "expected_tokens": 200
        },
        {
            "name": "OpenAI Reasoning Model (o1-mini)",
            "model_key": "openai:o1-mini",
            "prompt": "Solve this step by step: If a train travels 120 km in 2 hours, and another train travels 180 km in 3 hours, which train is faster and by how much?",
            "expected_tokens": 300
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['name']}")
        print("-" * 40)
        print(f"Prompt: {test_case['prompt']}")
        print(f"Expected tokens: ~{test_case['expected_tokens']}")
        print("\n🔄 Streaming output:")
        print("-" * 20)
        
        try:
            # Create streaming request
            response = generate(
                model_key=test_case["model_key"],
                input=test_case["prompt"],
                stream=True,
                max_output_tokens=test_case["expected_tokens"],
                temperature=0.7
            )
            
            # Process stream (llm-adapter returns different format)
            full_text = ""
            chunk_count = 0
            
            if hasattr(response, 'text'):
                # Non-streaming response
                print(response.text)
                full_text = response.text
            else:
                # Streaming response
                for chunk in response:
                    if hasattr(chunk, 'choices') and chunk.choices:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content'):
                            chunk_text = delta.content or ""
                            print(chunk_text, end="", flush=True)
                            full_text += chunk_text
                            chunk_count += 1
            
            print(f"\n📊 Stats:")
            print(f"   - Total chunks: {chunk_count}")
            print(f"   - Total characters: {len(full_text)}")
            print(f"   - Total words: {len(full_text.split())}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print(f"   Type: {type(e).__name__}")
            
        print("\n" + "=" * 50)


def test_openai_non_streaming():
    """Test OpenAI non-streaming for comparison."""
    
    print("\n\n🔍 Testing OpenAI Non-Streaming (Comparison)")
    print("=" * 50)
    
    handler = LLMHandler()
    
    try:
        response = handler.create(
            provider="openai",
            model="openai:fast",
            input="What is the capital of France? Give a brief explanation.",
            stream=False,
            max_output_tokens=100,
            temperature=0.3
        )
        
        print("📄 Non-streaming response:")
        print("-" * 30)
        print(response.output_text)
        
        print(f"\n📊 Response info:")
        print(f"   - Model: {response.model}")
        print(f"   - Usage: {response.usage}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def test_openai_embeddings():
    """Test OpenAI embeddings."""
    
    print("\n\n🧠 Testing OpenAI Embeddings")
    print("=" * 30)
    
    handler = LLMHandler()
    
    try:
        response = handler.create_embedding(
            provider="openai",
            model="text-embedding-3-small",
            input=["Hello world", "Goodbye world", "Artificial intelligence"]
        )
        
        print("✅ Embeddings generated successfully!")
        print(f"   - Number of embeddings: {len(response.data)}")
        print(f"   - Embedding dimension: {len(response.data[0].embedding)}")
        print(f"   - Usage: {response.usage}")
        
        # Show first few values of first embedding
        first_embedding = response.data[0].embedding
        print(f"   - First embedding sample: {first_embedding[:5]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def check_environment():
    """Check if required environment variables are set."""
    
    print("🔧 Environment Check")
    print("-" * 20)
    
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            masked = value[:8] + "*" * (len(value) - 8) if len(value) > 8 else "*" * len(value)
            print(f"✅ {var}: {masked}")
        else:
            print(f"❌ {var}: Not set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Please set them in your environment or .env file")
        return False
    
    return True


def main():
    """Main test function."""
    
    print("🧪 OpenAI LLM Handler Test Suite")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Exiting.")
        return
    
    print("\n✅ Environment check passed!")
    
    # Run tests
    try:
        test_openai_streaming()
        test_openai_non_streaming()
        test_openai_embeddings()
        
        print("\n\n🎉 All tests completed!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
