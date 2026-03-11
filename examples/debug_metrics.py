#!/usr/bin/env python3
"""
Debug Metrics Test - Single Call to Troubleshoot Token Accounting

This script makes a single session-based chat call to debug why metrics are empty.
"""

import requests
import json

def test_single_call():
    """Test a single session-based chat call and show all response data"""
    
    print("🔧 Debug Metrics Test - Single Call")
    print("="*60)
    
    # Step 1: Create session
    print("📝 Creating session...")
    session_response = requests.post("http://localhost:8000/chat/session")
    
    if session_response.status_code != 200:
        print(f"❌ Session creation failed: {session_response.status_code}")
        print(f"Response: {session_response.text}")
        return
    
    session_id = session_response.json()["session_id"]
    print(f"✅ Session created: {session_id}")
    
    # Step 2: Send one message
    print(f"\n📨 Sending message to session {session_id}...")
    
    payload = {
        "message": "What is 2+2?",
        "history": [],
        "params": {
            "top_k": 3,
            "temperature": 0.7,
            "max_output_tokens": 100
        }
    }
    
    response = requests.post(
        f"http://localhost:8000/chat/{session_id}",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=30
    )
    
    if response.status_code != 200:
        print(f"❌ Chat request failed: {response.status_code}")
        print(f"Response: {response.text}")
        return
    
    result = response.json()
    
    # Step 3: Show complete response structure
    print(f"\n🔍 Complete Response Structure:")
    print(f"Status: {response.status_code}")
    print(f"Response keys: {list(result.keys())}")
    
    print(f"\n📊 Full Response Data:")
    for key, value in result.items():
        print(f"   {key}: {value}")
    
    # Step 4: Show specific metrics fields
    print(f"\n🎯 Metrics Analysis:")
    
    print(f"\n📋 metrics field:")
    metrics = result.get("metrics", {})
    print(f"   Type: {type(metrics)}")
    print(f"   Value: {metrics}")
    print(f"   Keys: {list(metrics.keys()) if isinstance(metrics, dict) else 'N/A'}")
    
    print(f"\n📋 turn_metrics field:")
    turn_metrics = result.get("turn_metrics", {})
    print(f"   Type: {type(turn_metrics)}")
    print(f"   Value: {turn_metrics}")
    print(f"   Keys: {list(turn_metrics.keys()) if isinstance(turn_metrics, dict) else 'N/A'}")
    
    print(f"\n📋 conversation_totals field:")
    convo_totals = result.get("conversation_totals", {})
    print(f"   Type: {type(convo_totals)}")
    print(f"   Value: {convo_totals}")
    print(f"   Keys: {list(convo_totals.keys()) if isinstance(convo_totals, dict) else 'N/A'}")
    
    # Step 5: Show answer
    answer = result.get("answer", result.get("response", ""))
    print(f"\n💬 Answer: {answer}")
    
    print(f"\n🏁 Debug test completed!")

if __name__ == "__main__":
    test_single_call()
