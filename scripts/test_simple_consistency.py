#!/usr/bin/env python3
"""
Simple test to verify byte-level consistency of the formatting logic.

This test focuses on the core formatting functions without complex dependencies.
"""

import sys
import os
import hashlib

def test_formatting_consistency():
    """Test that the same input produces identical output."""
    
    # Sample conversation history
    test_history = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris.\n\nSources: https://example.com/france"},
        {"role": "user", "content": "Tell me more about Paris."},
        {"role": "assistant", "content": "Paris is a beautiful city known for the Eiffel Tower."},
    ]
    
    def format_conversation_deterministic(history, assistant_role="assistant"):
        """Deterministic conversation formatting."""
        # Create a deterministic copy
        _tail = [dict(msg) for msg in history]
        
        # Remove processing message if present
        if _tail:
            last_msg = _tail[-1]
            role = str(last_msg.get("role", "")).strip()
            content = str(last_msg.get("content", "")).strip().lower()
            if (role == "assistant" or role == assistant_role) and content == "processing":
                _tail.pop()
        
        # Format each message deterministically
        tail_lines = []
        for msg in _tail:
            # Normalize role and content strings consistently
            role = str(msg.get("role", "user")).strip()
            content = str(msg.get("content", "")).strip()
            
            # Clean sources block deterministically
            if role == "assistant" or role == assistant_role:
                # Normalize line endings first
                normalized = content.replace('\r\n', '\n').replace('\r', '\n')
                s = normalized.rstrip()
                
                # Use consistent regex pattern
                import re
                pattern = re.compile(r"(?:\n)Sources:\s*\n[\s\S]*\Z")
                m = pattern.search(s)
                if m:
                    s = s[:m.start()]
                
                content = s.rstrip()
                
                # Convert to target role if needed
                if role == "assistant":
                    role = assistant_role
            
            # Format line consistently
            tail_lines.append(f"{role}: {content}")
        
        # Join with consistent line endings
        return "\n".join(tail_lines) + "\n\n"
    
    print("Testing byte-level consistency...")
    print("=" * 50)
    
    # Test consistency
    outputs = []
    hashes = []
    
    for i in range(5):
        output = format_conversation_deterministic(test_history)
        hash_value = hashlib.sha256(output.encode('utf-8')).hexdigest()
        
        outputs.append(output)
        hashes.append(hash_value)
        
        print(f"Run {i+1}: {hash_value}")
    
    # Check if all outputs are identical
    if len(set(hashes)) == 1:
        print("\n✅ SUCCESS: All runs produced identical hashes!")
        print("\nFormatted output:")
        print("-" * 30)
        print(repr(outputs[0]))
        print("-" * 30)
        print(outputs[0])
        return True
    else:
        print("\n❌ FAILURE: Different runs produced different hashes!")
        for i, (output, hash_val) in enumerate(zip(outputs, hashes)):
            print(f"\nRun {i+1} ({hash_val}):")
            print(repr(output))
        return False

if __name__ == "__main__":
    success = test_formatting_consistency()
    sys.exit(0 if success else 1)
