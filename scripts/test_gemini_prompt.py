#!/usr/bin/env python3
"""
Test script to explore Gemini native SDK system instruction handling.
Tests how to properly separate system and user prompts for better system instruction compliance.
"""

import os
import sys
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment from {env_path}")
    else:
        print("⚠ .env file not found, using system environment")
except ImportError:
    print("⚠ python-dotenv not installed, using system environment only")

try:
    from google.genai import types
    from google.genai import Client as GeminiClient
except ImportError as e:
    print(f"Error: google-genai package not installed: {e}")
    print("Install with: pip install google-genai")
    sys.exit(1)

def test_current_flattened_approach():
    """Test the current flattened approach (like llm_handler.py does)."""
    print("=" * 60)
    print("TEST 1: Current Flattened Approach")
    print("=" * 60)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        return
    
    client = GeminiClient(api_key=api_key)
    
    system_prompt = (
        "You are a question-answering assistant for a retrieval-augmented system.\n"
        "STRICT RULES:\n"
        "1. Base your answer ONLY on information in the Context section.\n"
        "2. Do NOT use any outside knowledge, general world knowledge, training data, or assumptions beyond that context.\n"
        "3. If the context does not contain enough information to answer the question, reply with: I couldn't find any information to answer this question.\n"
        "4. Do not fabricate sources or facts.\n"
    )
    
    user_query = "What is the capital of France?"
    context = "Context: According to the provided documents, Paris is mentioned as a major European city."
    
    # Current approach: flatten everything
    flattened_prompt = f"system: {system_prompt}\n\n{context}\n\nuser: {user_query}"
    
    print(f"Flattened prompt:\n{flattened_prompt}\n")
    
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=flattened_prompt
        )
        
        print(f"Response:\n{response.text}\n")
        print("✓ Flattened approach completed\n")
        
    except Exception as e:
        print(f"Error with flattened approach: {e}\n")

def test_proper_system_instruction():
    """Test proper system_instruction separation."""
    print("=" * 60)
    print("TEST 2: Proper System Instruction Approach")
    print("=" * 60)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        return
    
    client = GeminiClient(api_key=api_key)
    
    system_prompt = (
        "You are a question-answering assistant for a retrieval-augmented system.\n"
        "STRICT RULES:\n"
        "1. Base your answer ONLY on information in the Context section.\n"
        "2. Do NOT use any outside knowledge, general world knowledge, training data, or assumptions beyond that context.\n"
        "3. If the context does not contain enough information to answer the question, reply with: I couldn't find any information to answer this question.\n"
        "4. Do not fabricate sources or facts.\n"
    )
    
    user_query = "What is the capital of France?"
    context = "Context: According to the provided documents, Paris is mentioned as a major European city."
    
    # Proper approach: separate system instruction and user content
    user_content = f"{context}\n\nQuestion: {user_query}"
    
    print(f"System instruction:\n{system_prompt}\n")
    print(f"User content:\n{user_content}\n")
    
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt
            )
        )
        
        print(f"Response:\n{response.text}\n")
        print("✓ System instruction approach completed\n")
        
    except Exception as e:
        print(f"Error with system instruction approach: {e}\n")

def test_messages_to_text_conversion():
    """Test the _messages_to_text function from llm_handler.py."""
    print("=" * 60)
    print("TEST 3: Messages-to-Text Conversion")
    print("=" * 60)
    
    def _messages_to_text(messages):
        """Replicate the function from llm_handler.py"""
        if not isinstance(messages, list):
            return str(messages)
        parts = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "")
            content = m.get("content")
            if isinstance(content, list):
                joined = []
                for c in content:
                    if isinstance(c, dict) and "text" in c:
                        joined.append(str(c.get("text") or ""))
                    else:
                        joined.append(str(c))
                content_s = "".join(joined)
            else:
                content_s = "" if content is None else str(content)
            if role:
                parts.append(f"{role}: {content_s}")
            else:
                parts.append(content_s)
        return "\n".join([p for p in parts if p is not None])
    
    # Simulate the messages structure from chat_manager.py
    system_prompt = (
        "You are a question-answering assistant for a retrieval-augmented system.\n"
        "STRICT RULES:\n"
        "1. Base your answer ONLY on information in the Context section.\n"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": "Context: According to the provided documents, Paris is mentioned as a major European city."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    converted = _messages_to_text(messages)
    print(f"Original messages:\n{messages}\n")
    print(f"Converted text:\n{converted}\n")
    print("✓ Messages-to-text conversion completed\n")

def test_with_responses_create_signature():
    """Test using the same call signature as _responses_create."""
    print("=" * 60)
    print("TEST 5: Using _responses_create Call Signature")
    print("=" * 60)
    
    # Simulate the call signature: _responses_create(provider=..., model=..., input=..., **kwargs)
    def mock_responses_create(provider: str | None = None, **kwargs):
        """Mock version of _responses_create for testing"""
        print(f"Called with provider={provider}")
        print(f"kwargs keys: {list(kwargs.keys())}")
        
        model = kwargs.get("model")
        inp = kwargs.get("input")
        stream = kwargs.get("stream", False)
        
        print(f"model: {model}")
        print(f"input type: {type(inp)}")
        print(f"stream: {stream}")
        
        if isinstance(inp, list):
            print(f"Input messages: {inp}")
            # Show how this would be processed by llm_handler
            print("→ This would go to llm_handler.create() with messages list")
        else:
            print(f"Input string: {inp[:100]}...")
            print("→ This would go to llm_handler.create() with flattened string")
        
        return {"mock": "response"}
    
    # Test with messages list (current chat_manager.py behavior for tools)
    print("Scenario A: Messages list (tools enabled)")
    messages_input = [
        {"role": "system", "content": "You are a RAG assistant. Answer ONLY from context."},
        {"role": "system", "content": "Context: Paris is mentioned as a major European city."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    result_a = mock_responses_create(
        provider="gemini",
        model="gemini-3-flash-preview",
        input=messages_input,
        stream=False,
        temperature=0.7
    )
    
    print()
    
    # Test with flattened string (current behavior for non-tools)
    print("Scenario B: Flattened string (no tools)")
    flattened_input = "system: You are a RAG assistant. Answer ONLY from context.\n\nContext: Paris is mentioned as a major European city.\n\nuser: What is the capital of France?"
    
    result_b = mock_responses_create(
        provider="gemini", 
        model="gemini-3-flash-preview",
        input=flattened_input,
        stream=False,
        temperature=0.7
    )
    
    print()
    print("✓ _responses_create signature testing completed\n")

def test_proposed_responses_create_fix():
    """Test how _responses_create could be modified for proper system instruction handling."""
    print("=" * 60)
    print("TEST 6: Proposed _responses_create Fix")
    print("=" * 60)
    
    def improved_responses_create(provider: str | None = None, **kwargs):
        """Improved version that handles system instructions properly for Gemini"""
        prov = (provider or "openai").strip().lower()
        
        if provider is None or prov == "openai":
            # OpenAI path - no changes needed
            return "OpenAI: Would call llm_handler.responses.create(**kwargs)"
        
        # Enhanced Gemini path
        model = kwargs.pop("model", None)
        inp = kwargs.pop("input", None)
        stream = bool(kwargs.pop("stream", False))
        
        # NEW: Extract system instruction for Gemini native SDK
        if prov == "gemini" and isinstance(inp, list):
            system_parts = []
            user_parts = []
            
            for m in inp:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role") or "")
                content = m.get("content", "")
                
                if role == "system":
                    system_parts.append(str(content))
                else:
                    user_parts.append(f"{role}: {content}" if role else str(content))
            
            if system_parts:
                # Modify kwargs to include system_instruction
                kwargs["system_instruction"] = "\n\n".join(system_parts)
                # Replace input with only user content
                inp = "\n\n".join(user_parts) if user_parts else ""
        
        return f"Gemini: Would call llm_handler.create(provider={prov}, model={model}, input={type(inp)}, system_instruction={'✓' if 'system_instruction' in kwargs else '✗'}, **kwargs)"
    
    # Test the improved version
    messages_input = [
        {"role": "system", "content": "You are a RAG assistant. Answer ONLY from context."},
        {"role": "system", "content": "Context: Paris is mentioned as a major European city."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    result = improved_responses_create(
        provider="gemini",
        model="gemini-3-flash-preview", 
        input=messages_input,
        stream=False,
        temperature=0.7
    )
    
    print(f"Improved call result: {result}\n")
    print("✓ Proposed _responses_create fix completed\n")

def test_proposed_llm_handler_fix():
    """Test how the proposed fix would work."""
    print("=" * 60)
    print("TEST 4: Proposed LLM Handler Fix")
    print("=" * 60)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        return
    
    client = GeminiClient(api_key=api_key)
    
    def extract_system_and_user_content(messages):
        """Proposed function to separate system and user content"""
        system_parts = []
        user_parts = []
        
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "")
            content = m.get("content", "")
            
            if role == "system":
                system_parts.append(str(content))
            else:
                user_parts.append(f"{role}: {content}" if role else str(content))
        
        system_instruction = "\n\n".join(system_parts) if system_parts else None
        user_content = "\n\n".join(user_parts) if user_parts else None
        
        return system_instruction, user_content
    
    # Test messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant that speaks like a 17th-century pirate."},
        {"role": "system", "content": "Context: According to the provided documents, Paris is mentioned as a major European city."},
        {"role": "user", "content": "What be the capital of France?"}
    ]
    
    system_instruction, user_content = extract_system_and_user_content(messages)
    
    print(f"Extracted system instruction:\n{system_instruction}\n")
    print(f"Extracted user content:\n{user_content}\n")
    
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        
        print(f"Response:\n{response.text}\n")
        print("✓ Proposed fix completed\n")
        
    except Exception as e:
        print(f"Error with proposed fix: {e}\n")

def test_gemini_openai_adapter_path():
    """Test the Gemini OpenAI adapter path (not native SDK)."""
    print("=" * 60)
    print("TEST 7: Gemini OpenAI Adapter Path")
    print("=" * 60)
    
    # This simulates the current llm_handler.py Gemini adapter path
    # which uses OpenAI client pointed at Gemini OpenAI adapter endpoint
    
    api_key = os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("GEMINI_OPENAI_BASE_URL")
    
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        return
    
    if not base_url:
        print("Error: GEMINI_OPENAI_BASE_URL not set")
        print("This is needed for the OpenAI adapter path")
        return
    
    try:
        from openai import OpenAI
        
        # Create OpenAI client pointed at Gemini adapter (like llm_handler.py does)
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        system_prompt = (
            "You are a question-answering assistant for a retrieval-augmented system.\n"
            "STRICT RULES:\n"
            "1. Base your answer ONLY on information in the Context section.\n"
            "2. Do NOT use any outside knowledge, general world knowledge, training data, or assumptions beyond that context.\n"
            "3. If the context does not contain enough information to answer the question, reply with: I couldn't find any information to answer this question.\n"
            "4. Do not fabricate sources or facts.\n"
        )
        
        user_query = "What is the capital of France?"
        context = "Context: According to the provided documents, Paris is mentioned as a major European city."
        
        # Test 7A: Combined system message (recommended approach)
        print("Scenario A: Combined system message (recommended approach)")
        combined_system_prompt = f"{system_prompt}\n\n{context}"
        messages = [
            {"role": "system", "content": combined_system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        print(f"Combined system prompt: {combined_system_prompt}\n")
        print(f"Messages: {messages}\n")
        
        try:
            response = client.chat.completions.create(
                model="gemini-3-flash-preview",
                messages=messages,
                temperature=0.7
            )
            
            print(f"Response (combined system message):\n{response.choices[0].message.content}\n")
            print("✓ Gemini OpenAI adapter (combined system) completed\n")
            
        except Exception as e:
            print(f"Error with combined system message: {e}\n")
        
        # Test 7B: Flattened format (alternative)
        print("Scenario B: Flattened format (like current non-tools path)")
        flattened_prompt = f"system: {system_prompt}\n\n{context}\n\nuser: {user_query}"
        
        print(f"Flattened prompt: {flattened_prompt[:200]}...\n")
        
        try:
            response = client.chat.completions.create(
                model="gemini-3-flash-preview", 
                messages=[{"role": "user", "content": flattened_prompt}],
                temperature=0.7
            )
            
            print(f"Response (flattened format):\n{response.choices[0].message.content}\n")
            print("✓ Gemini OpenAI adapter (flattened) completed\n")
            
        except Exception as e:
            print(f"Error with flattened format: {e}\n")
            
    except ImportError:
        print("Error: openai package not installed")
        print("Install with: pip install openai")
    except Exception as e:
        print(f"Error setting up Gemini OpenAI adapter: {e}")

def test_gemini_adapter_vs_native_comparison():
    """Compare Gemini OpenAI adapter vs Native SDK system instruction handling."""
    print("=" * 60)
    print("TEST 8: Gemini Adapter vs Native SDK Comparison")
    print("=" * 60)
    
    system_prompt = (
        "You are a helpful assistant that speaks like a 17th-century pirate. "
        "You must ONLY answer using information from the provided context. "
        "If the context doesn't contain the answer, say 'Arrr, I can't find that treasure in me maps!'"
    )
    
    context = "Context: According to the provided documents, Paris is mentioned as a major European city known for the Eiffel Tower."
    question = "What is the capital of France?"
    
    # Test 8A: Native SDK with proper system instruction
    print("Scenario A: Native SDK with system_instruction")
    try:
        from google.genai import types
        from google.genai import Client as GeminiClient
        
        client = GeminiClient(api_key=os.getenv("GEMINI_API_KEY"))
        
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=f"{context}\n\nQuestion: {question}",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt
            )
        )
        
        print(f"Native SDK Response:\n{response.text}\n")
        print("✓ Native SDK completed\n")
        
    except Exception as e:
        print(f"Native SDK error: {e}\n")
    
    # Test 8B: OpenAI adapter with combined system message
    print("Scenario B: OpenAI adapter with combined system message")
    try:
        from openai import OpenAI
        
        base_url = os.getenv("GEMINI_OPENAI_BASE_URL")
        if not base_url:
            print("Skipping OpenAI adapter test - GEMINI_OPENAI_BASE_URL not set\n")
            return
            
        client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url=base_url
        )
        
        combined_system_prompt = f"{system_prompt}\n\n{context}"
        messages = [
            {"role": "system", "content": combined_system_prompt},
            {"role": "user", "content": question}
        ]
        
        response = client.chat.completions.create(
            model="gemini-3-flash-preview",
            messages=messages,
            temperature=0.7
        )
        
        print(f"OpenAI Adapter Response:\n{response.choices[0].message.content}\n")
        print("✓ OpenAI adapter completed\n")
        
    except Exception as e:
        print(f"OpenAI adapter error: {e}\n")
    
    print("Comparison: Native SDK should have better system instruction compliance than OpenAI adapter")

def main():
    print("Gemini System Instruction Testing")
    print("================================")
    print(f"Using model: gemini-3-flash-preview")
    print(f"API Key: {'✓ Set' if os.getenv('GEMINI_API_KEY') else '✗ Missing'}")
    print(f"OpenAI Base URL: {'✓ Set' if os.getenv('GEMINI_OPENAI_BASE_URL') else '✗ Missing'}")
    print()
    
    # Run all tests
    #test_current_flattened_approach()
    #test_proper_system_instruction()
    #test_messages_to_text_conversion()
    test_proposed_llm_handler_fix()
    #test_with_responses_create_signature()
    test_proposed_responses_create_fix()
    test_gemini_openai_adapter_path()
    test_gemini_adapter_vs_native_comparison()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("1. Current approach flattens system prompts into regular text")
    print("2. Proper system_instruction provides better compliance")
    print("3. Proposed fix would extract system messages separately")
    print("4. _responses_create signature can be enhanced for Gemini")
    print("5. OpenAI adapter path may have different system instruction behavior")
    print("6. Native SDK vs OpenAI adapter comparison shows compliance differences")
    print("7. This explains why Gemini may be less compliant than OpenAI")

if __name__ == "__main__":
    main()
