#!/usr/bin/env python3
"""
Session-Based Chat Demo with Token Accounting

This script demonstrates the stateful chat API with:
- Session creation and management
- Multi-turn conversation with context preservation
- Token accounting across multiple messages
- Model override capabilities
- Session history verification

Usage:
    python examples/session_stateful_chat.py
"""

import requests
import json
import time
from typing import Dict, Any, Optional


class SessionChatDemo:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id: Optional[str] = None
        self.message_count = 0
        
    def create_session(self) -> str:
        """Create a new chat session"""
        print("🔧 Creating new session...")
        response = requests.post(f"{self.base_url}/chat/session")
        
        if response.status_code == 200:
            self.session_id = response.json()["session_id"]
            print(f"✅ Session created: {self.session_id}")
            return self.session_id
        else:
            print(f"❌ Failed to create session: {response.status_code}")
            print(f"Response: {response.text}")
            raise Exception("Session creation failed")
    
    def send_message(
        self, 
        message: str, 
        model_keys: Optional[Dict[str, str]] = None,
        max_output_tokens: int = 500
    ) -> Dict[str, Any]:
        """Send a message to the session and return response with token usage"""
        if not self.session_id:
            raise Exception("No active session. Call create_session() first.")
        
        self.message_count += 1
        print(f"\n📨 Message {self.message_count}: {message}")
        
        # Prepare request payload
        payload = {
            "message": message,
            "history": [],  # Not needed for session-based chat
            "params": {
                "top_k": 5,
                "temperature": 0.7,
                "max_output_tokens": max_output_tokens
            }
        }
        
        # Add model overrides if specified
        if model_keys:
            payload["params"]["model_keys"] = model_keys
            print(f"🤖 Using model overrides:")
            for stage, model in model_keys.items():
                print(f"   - {stage}: {model}")
        
        # Send request
        response = requests.post(
            f"{self.base_url}/chat/{self.session_id}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            self._print_response(result)
            return result
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            raise Exception(f"Chat request failed: {response.status_code}")
    
    def get_session_history(self) -> Dict[str, Any]:
        """Get the complete session history"""
        if not self.session_id:
            raise Exception("No active session")
        
        print(f"\n📚 Getting session history...")
        response = requests.get(f"{self.base_url}/chat/{self.session_id}/history")
        
        if response.status_code == 200:
            history = response.json()
            print(f"📝 Session has {len(history['messages'])} messages")
            return history
        else:
            print(f"❌ Failed to get history: {response.status_code}")
            return {}
    
    def _print_response(self, result: Dict[str, Any]) -> None:
        """Print response with detailed token usage information"""
        print("\n" + "="*60)
        print("🤖 RESPONSE")
        print("="*60)
        
        # Answer
        answer = result.get("answer", result.get("response", ""))
        print(f"\n💬 Answer:\n{answer}")
        
        # Debug: Show raw metrics structure
        print(f"\n🔍 Raw Metrics Data:")
        print(f"   metrics: {result.get('metrics', {})}")
        print(f"   turn_metrics: {result.get('turn_metrics', {})}")
        print(f"   conversation_totals: {result.get('conversation_totals', {})}")
        
        # Token usage - detailed breakdown
        if "turn_metrics" in result and result["turn_metrics"]:
            print(f"\n📊 Current Turn Metrics:")
            turn_metrics = result["turn_metrics"]
            
            # Extract token and cost data from turn_metrics
            if isinstance(turn_metrics, dict):
                # Show stage-specific metrics
                for stage, data in turn_metrics.items():
                    if isinstance(data, dict) and stage in ["embedding", "rewrite", "rerank", "inference", "summary"]:
                        print(f"   🪙 {stage.capitalize():12s}:")
                        if "input_tokens" in data:
                            print(f"      - Input tokens : {data['input_tokens']:6,}")
                        if "output_tokens" in data:
                            print(f"      - Output tokens: {data['output_tokens']:6,}")
                        if "cost" in data:
                            print(f"      - Cost        : ${data['cost']:8.6f}")
                        # For embedding, might only have input_tokens
                        if "input_tokens" not in data and "output_tokens" not in data and "tokens" in data:
                            print(f"      - Tokens      : {data['tokens']:6,}")
                
                # Calculate totals
                total_input = sum(data.get("input_tokens", 0) for data in turn_metrics.values() if isinstance(data, dict))
                total_output = sum(data.get("output_tokens", 0) for data in turn_metrics.values() if isinstance(data, dict))
                total_cost = sum(data.get("cost", 0) for data in turn_metrics.values() if isinstance(data, dict))
                
                print(f"   🪙 {'TOTAL':12s}:")
                print(f"      - Input tokens : {total_input:6,}")
                print(f"      - Output tokens: {total_output:6,}")
                print(f"      - Total tokens : {total_input + total_output:6,}")
                print(f"      - Total cost   : ${total_cost:8.6f}")
        
        # Also show legacy metrics if available
        if "metrics" in result and result["metrics"]:
            print(f"\n📊 Legacy Metrics:")
            legacy = result["metrics"]
            for key, value in legacy.items():
                print(f"   - {key}: {value}")
        
        # Turn metrics (additional details)
        if "turn_metrics" in result and result["turn_metrics"]:
            print(f"\n🔄 Additional Turn Metrics:")
            turn_metrics = result["turn_metrics"]
            for key, value in turn_metrics.items():
                if key not in ["tokens", "cost"]:  # Already shown above
                    print(f"   - {key}: {value}")
        
        # Conversation totals
        if "conversation_totals" in result:
            print(f"\n📈 Conversation Totals:")
            totals = result["conversation_totals"]
            
            if "tokens" in totals:
                tokens = totals["tokens"]
                total_tokens = tokens.get("total", 0)
                print(f"   🪙 Cumulative Tokens: {total_tokens:,}")
                
                # Detailed breakdown if available
                if isinstance(tokens, dict) and len(tokens) > 1:
                    print(f"   📊 Stage Totals:")
                    for stage, count in tokens.items():
                        if stage != "total":
                            print(f"      - {stage:12s}: {count:6,}")
            
            if "cost" in totals:
                cost = totals["cost"]
                total_cost = cost.get("total", 0)
                print(f"   💰 Cumulative Cost: ${total_cost:.6f}")
                
                # Detailed breakdown if available
                if isinstance(cost, dict) and len(cost) > 1:
                    print(f"   💸 Cost Totals:")
                    for stage, amount in cost.items():
                        if stage != "total":
                            print(f"      - {stage:12s}: ${amount:8.6f}")
            
            if "messages" in totals:
                print(f"   📨 Messages: {totals['messages']}")
                
                # Calculate average tokens per message
                if "tokens" in totals and totals["tokens"].get("total", 0) > 0:
                    avg_tokens = totals["tokens"]["total"] / totals["messages"]
                    print(f"   📊 Avg Tokens/Message: {avg_tokens:.1f}")
        
        # Additional conversation totals details
        if "conversation_totals" in result and result["conversation_totals"]:
            totals = result["conversation_totals"]
            other_metrics = {k: v for k, v in totals.items() 
                           if k not in ["tokens", "cost", "messages"]}
            if other_metrics:
                print(f"\n📋 Other Conversation Metrics:")
                for key, value in other_metrics.items():
                    print(f"   - {key}: {value}")
        
        # Tools used
        if "tools_used" in result and result["tools_used"]:
            print(f"\n🔧 Tools Used: {result['tools_used']}")
        
        # Sources
        if "sources" in result and result["sources"]:
            print(f"\n📚 Sources: {len(result['sources'])} sources found")
            for i, source in enumerate(result["sources"][:3], 1):
                title = source.get("title", "Unknown")[:50]
                url = source.get("url", "")[:50]
                print(f"   {i}. {title} ({url})")
        
        print("="*60)
    
    def run_demo(self) -> None:
        """Run a complete demo session with various model configurations"""
        print("🚀 Starting Session-Based Chat Demo")
        print("="*60)
        
        try:
            # Create session
            self.create_session()
            
            # Message 1: Basic question (default models)
            self.send_message("What is Mount Everest?")
            
            # Message 2: Follow-up (tests context preservation)
            self.send_message("How tall is it?")
            
            # Message 3: Complex follow-up (tests deeper context)
            self.send_message("What is the elevation difference with Kilimanjaro?")
            
            # Message 4: Model override - Gemini for inference only
            self.send_message(
                "Explain the climbing routes on Everest in simple terms",
                model_keys={"inference": "gemini:openai-2.5-flash-lite"}
            )
            
            # Message 5: Multiple stage overrides
            self.send_message(
                "Compare Everest with other high peaks",
                model_keys={
                    "inference": "gemini:openai-2.5-flash-lite",
                    "rewrite": "openai:gpt-4o-mini",
                    "rerank": "openai:gpt-4o-mini",
                    "summary": "openai:gpt-4o-mini"
                }
            )
            
            # Message 6: OpenAI models comparison
            self.send_message(
                "What are the main dangers of climbing Everest?",
                model_keys={
                    "inference": "openai:gpt-4o",
                    "rewrite": "openai:gpt-4o-mini"
                }
            )
            
            # Get session history
            history = self.get_session_history()
            
            # Final summary
            print(f"\n🎉 Demo completed!")
            print(f"📊 Session Summary:")
            print(f"   Session ID: {self.session_id}")
            print(f"   Total Messages: {len(history.get('messages', []))}")
            print(f"   Session Duration: Active")
            print(f"   Model Configurations Tested:")
            print(f"     - Default models")
            print(f"     - Gemini inference only")
            print(f"     - Multi-stage overrides")
            print(f"     - OpenAI models")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
        
        print("\n🏁 Demo finished")


def main():
    """Main entry point"""
    print("🎯 Session-Based Chat Demo with Model Override Examples")
    print("="*60)
    print("\n📋 Model Configuration Examples:")
    print("1. Default models (no overrides)")
    print("2. Single stage override: {'inference': 'gemini:openai-2.5-flash-lite'}")
    print("3. Multi-stage overrides:")
    print("   {'inference': 'gemini:openai-2.5-flash-lite',")
    print("    'rewrite': 'openai:gpt-4o-mini',")
    print("    'rerank': 'openai:gpt-4o-mini',")
    print("    'summary': 'openai:gpt-4o-mini'}")
    print("4. OpenAI models: {'inference': 'openai:gpt-4o', 'rewrite': 'openai:gpt-4o-mini'}")
    print("\n📊 Watch token usage accumulate across messages!")
    print("="*60)
    
    demo = SessionChatDemo()
    demo.run_demo()


def demo_model_configurations():
    """Demonstrate different model_keys configurations"""
    demo = SessionChatDemo()
    
    print("🎯 Testing Different Model Configurations")
    print("="*60)
    
    try:
        demo.create_session()
        
        # Test 1: Default models
        print("\n1️⃣ DEFAULT MODELS")
        demo.send_message("What is artificial intelligence?")
        
        # Test 2: Gemini inference only
        print("\n2️⃣ GEMINI INFERENCE ONLY")
        demo.send_message(
            "Explain AI in simple terms",
            model_keys={"inference": "gemini:openai-2.5-flash-lite"}
        )
        
        # Test 3: Multi-stage configuration
        print("\n3️⃣ MULTI-STAGE CONFIGURATION")
        demo.send_message(
            "Compare machine learning and deep learning",
            model_keys={
                "inference": "gemini:openai-2.5-flash-lite",
                "rewrite": "openai:gpt-4o-mini",
                "rerank": "openai:gpt-4o-mini"
            }
        )
        
        # Test 4: All OpenAI
        print("\n4️⃣ ALL OPENAI MODELS")
        demo.send_message(
            "What are the latest AI trends?",
            model_keys={
                "inference": "openai:gpt-4o",
                "rewrite": "openai:gpt-4o-mini",
                "rerank": "openai:gpt-4o-mini",
                "summary": "openai:gpt-4o-mini"
            }
        )
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--models":
        demo_model_configurations()
    else:
        main()
