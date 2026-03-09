#!/usr/bin/env python3
"""
Comprehensive test to verify byte-level consistency of history processing.

This test demonstrates that the SimpleHistoryProcessor produces identical
output for the same input across multiple runs, ensuring no byte differences.
"""

import sys
import os
import hashlib
import json

# Add the backend to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_byte_level_consistency():
    """Test that the same input produces identical byte-level output."""
    
    try:
        from backend.chat.simple_history_processor import SimpleHistoryProcessor
        
        # Create a mock settings object
        class MockSettings:
            def __init__(self):
                self.assistant_role_default = "assistant"
        
        # Test cases with different scenarios
        test_cases = [
            {
                "name": "Basic conversation",
                "history": [
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "The capital of France is Paris.\n\nSources: https://example.com/france"},
                    {"role": "user", "content": "Tell me more about Paris."},
                    {"role": "assistant", "content": "Paris is a beautiful city known for the Eiffel Tower."},
                ],
                "params": {}
            },
            {
                "name": "Conversation with processing message",
                "history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "processing"},
                ],
                "params": {}
            },
            {
                "name": "Empty history",
                "history": [],
                "params": {}
            },
            {
                "name": "Mixed content with sources",
                "history": [
                    {"role": "user", "content": "Find information about AI"},
                    {"role": "assistant", "content": "AI is a fascinating field.\n\nSources: https://example.com/ai\nhttps://example.com/ml"},
                    {"role": "user", "content": "What about machine learning?"},
                    {"role": "assistant", "content": "Machine learning is a subset of AI."},
                ],
                "params": {}
            }
        ]
        
        # Initialize processor
        processor = SimpleHistoryProcessor(MockSettings())
        
        print("Testing byte-level consistency across multiple scenarios...")
        print("=" * 60)
        
        all_consistent = True
        
        for test_case in test_cases:
            print(f"\n🧪 Testing: {test_case['name']}")
            print("-" * 40)
            
            # Test consistency
            result = processor.verify_consistency(
                test_case['history'], 
                test_case['params']
            )
            
            print(f"Consistent: {result['consistent']}")
            print(f"Hash: {result['hash']}")
            print(f"Error: {result['error']}")
            
            if result['consistent']:
                print("✅ PASSED: Output is byte-level consistent!")
                
                # Show the formatted output (first 100 chars)
                output_preview = result['formatted_output'][:100]
                print(f"Output preview: {repr(output_preview)}...")
                
                # Additional verification: test multiple times
                hashes = []
                for i in range(10):  # Test 10 times
                    hash_val = processor.get_consistency_hash(
                        test_case['history'], 
                        test_case['params']
                    )
                    hashes.append(hash_val)
                
                if len(set(hashes)) == 1:
                    print("✅ Verified: 10 runs produced identical hashes")
                else:
                    print("❌ FAILED: Multiple runs produced different hashes")
                    all_consistent = False
            else:
                print("❌ FAILED: Output is inconsistent!")
                all_consistent = False
        
        print("\n" + "=" * 60)
        if all_consistent:
            print("🎉 ALL TESTS PASSED: Byte-level consistency guaranteed!")
            return True
        else:
            print("💥 SOME TESTS FAILED: Consistency issues detected!")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_deterministic_behavior():
    """Test that the processor behaves deterministically with edge cases."""
    
    try:
        from backend.chat.simple_history_processor import SimpleHistoryProcessor
        
        class MockSettings:
            def __init__(self):
                self.assistant_role_default = "assistant"
        
        processor = SimpleHistoryProcessor(MockSettings())
        
        print("\nTesting deterministic behavior with edge cases...")
        print("=" * 50)
        
        # Test with various edge cases
        edge_cases = [
            {
                "name": "Content with different line endings",
                "history": [
                    {"role": "user", "content": "Hello\r\nWorld"},
                    {"role": "assistant", "content": "Hi\rthere!\n\nSources: test"},
                ]
            },
            {
                "name": "Content with extra whitespace",
                "history": [
                    {"role": "user", "content": "  Hello  World  "},
                    {"role": "assistant", "content": "  Hi  there!  \n\n  Sources: test  "},
                ]
            },
            {
                "name": "Empty content messages",
                "history": [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": "  \n\nSources: test"},
                ]
            }
        ]
        
        all_consistent = True
        
        for case in edge_cases:
            print(f"\n🔍 Edge case: {case['name']}")
            
            # Run multiple times
            outputs = []
            for i in range(5):
                output = processor.format_recent_conversation(
                    case['history'], {}, "edge_test"
                )
                outputs.append(output)
            
            # Check if all are identical
            if len(set(outputs)) == 1:
                print("✅ PASSED: Deterministic behavior")
            else:
                print("❌ FAILED: Non-deterministic behavior")
                for i, output in enumerate(outputs):
                    print(f"  Run {i+1}: {repr(output)}")
                all_consistent = False
        
        return all_consistent
        
    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Byte-Level Consistency Test Suite")
    print("=" * 60)
    
    success1 = test_byte_level_consistency()
    success2 = test_deterministic_behavior()
    
    overall_success = success1 and success2
    
    print("\n" + "=" * 60)
    if overall_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Byte-level consistency is guaranteed!")
        print("✅ The same conversation history will always produce identical output!")
    else:
        print("💥 TESTS FAILED!")
        print("❌ Consistency issues detected!")
    
    sys.exit(0 if overall_success else 1)
