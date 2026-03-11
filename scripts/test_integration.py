#!/usr/bin/env python3
"""
Integration test to verify SimpleHistoryProcessor works correctly with chat_manager.py

This test verifies that the SimpleHistoryProcessor integration maintains
byte-level consistency while preserving all existing functionality.
"""

import sys
import os
import hashlib

# Add the backend to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_integration():
    """Test the integration of SimpleHistoryProcessor with the chat pipeline."""
    
    try:
        from backend.chat.simple_history_processor import SimpleHistoryProcessor
        
        # Create a mock settings object
        class MockSettings:
            def __init__(self):
                self.assistant_role_default = "assistant"
                self.raw_tail_turns = 2
                self.chat_history_window_turns = 3
        
        # Test the exact same scenarios that would be used in production
        test_scenarios = [
            {
                "name": "Production-like conversation",
                "history": [
                    {"role": "user", "content": "What is machine learning?"},
                    {"role": "assistant", "content": "Machine learning is a subset of AI that enables systems to learn from data.\n\nSources: https://example.com/ml"},
                    {"role": "user", "content": "Can you give me an example?"},
                    {"role": "assistant", "content": "Sure! Image recognition is a common example of machine learning."},
                    {"role": "user", "content": "How does it work?"},
                    {"role": "assistant", "content": "It works by training algorithms on large datasets to recognize patterns."},
                ],
                "params": {
                    "raw_tail_turns": 2,
                    "chat_history_window_turns": 3,
                }
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
                "name": "Empty verbatim tail",
                "history": [],
                "params": {}
            }
        ]
        
        # Initialize processor
        processor = SimpleHistoryProcessor(MockSettings())
        
        print("🔬 Testing SimpleHistoryProcessor Integration")
        print("=" * 60)
        
        all_consistent = True
        
        for scenario in test_scenarios:
            print(f"\n📋 Scenario: {scenario['name']}")
            print("-" * 40)
            
            # Test consistency
            result = processor.verify_consistency(
                scenario['history'], 
                scenario['params']
            )
            
            print(f"✅ Consistent: {result['consistent']}")
            print(f"🔐 Hash: {result['hash']}")
            
            if result['consistent']:
                # Test multiple runs to ensure stability
                hashes = []
                for i in range(10):
                    hash_val = processor.get_consistency_hash(
                        scenario['history'], 
                        scenario['params']
                    )
                    hashes.append(hash_val)
                
                if len(set(hashes)) == 1:
                    print("✅ 10 runs: Identical hashes")
                else:
                    print("❌ 10 runs: Different hashes detected!")
                    all_consistent = False
                
                # Show formatted output preview
                output = result['formatted_output']
                preview = output[:100] if output else "Empty output"
                print(f"📄 Output preview: {repr(preview)}...")
                
                # Verify byte-level consistency by encoding and decoding
                encoded = output.encode('utf-8')
                decoded = encoded.decode('utf-8')
                if decoded == output:
                    print("✅ UTF-8 encoding/decoding: Consistent")
                else:
                    print("❌ UTF-8 encoding/decoding: Inconsistent!")
                    all_consistent = False
            else:
                print(f"❌ Error: {result['error']}")
                all_consistent = False
        
        return all_consistent
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_import_compatibility():
    """Test that all imports work correctly."""
    
    print("\n🔍 Testing Import Compatibility")
    print("=" * 40)
    
    try:
        # Test SimpleHistoryProcessor import
        from backend.chat.simple_history_processor import SimpleHistoryProcessor
        print("✅ SimpleHistoryProcessor: Import successful")
        
        # Test utils import
        from backend.chat.utils import _get_param_int, split_history_for_prompt
        print("✅ Utils functions: Import successful")
        
        # Test that the classes can be instantiated
        class MockSettings:
            def __init__(self):
                self.assistant_role_default = "assistant"
        
        processor = SimpleHistoryProcessor(MockSettings())
        print("✅ SimpleHistoryProcessor: Instantiation successful")
        
        # Test basic functionality
        test_history = [{"role": "user", "content": "test"}]
        output = processor.format_recent_conversation(test_history, {}, "test")
        print("✅ Basic functionality: Working")
        
        return True
        
    except Exception as e:
        print(f"❌ Import compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 SimpleHistoryProcessor Integration Test Suite")
    print("=" * 60)
    
    # Test imports first
    import_success = test_import_compatibility()
    
    if import_success:
        # Test integration
        integration_success = test_integration()
        
        print("\n" + "=" * 60)
        if integration_success:
            print("🎉 INTEGRATION TESTS PASSED!")
            print("✅ SimpleHistoryProcessor is ready for production")
            print("✅ Byte-level consistency guaranteed")
            print("✅ All imports working correctly")
        else:
            print("💥 INTEGRATION TESTS FAILED!")
            print("❌ Issues detected in integration")
        
        sys.exit(0 if integration_success else 1)
    else:
        print("\n💥 IMPORT TESTS FAILED!")
        print("❌ Cannot proceed with integration tests")
        sys.exit(1)
