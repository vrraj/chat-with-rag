#!/usr/bin/env python3

import os
import sys
from typing import Any


def main() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Import module so we can monkeypatch its llm_handler binding.
    import backend.chat.chunked_history_manager as chm

    # --- Fake LLM handler (no API calls) ---
    class _FakeResponses:
        def create(self, *, model: str, input: Any, stream: bool = False, **kwargs: Any):
            # Return an object that build_llm_result_from_response can consume via our fake.
            return {"_fake": True, "model": model, "input": input, "kwargs": kwargs}

    class _FakeLLMHandler:
        def __init__(self):
            self.responses = _FakeResponses()

        def build_llm_result_from_response(self, resp: Any, *, provider: str = "openai"):
            # Produce deterministic summary text.
            # Include the number of messages in the recent chunk to prove rollover was called.
            try:
                inp = (resp or {}).get("input") or []
                # count user+assistant lines in user payload (best-effort)
                user_msg = ""
                for m in inp:
                    if isinstance(m, dict) and m.get("role") == "user":
                        user_msg = str(m.get("content") or "")
                        break
                msg_count = user_msg.count("\n") + 1 if user_msg.strip() else 0
            except Exception:
                msg_count = 0
            return {"text": f"UPDATED_SUMMARY(msg_lines={msg_count})", "usage": {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}}

    # Patch the module-level llm_handler used by ChunkedHistoryManager.
    chm.llm_handler = _FakeLLMHandler()

    # --- Minimal settings stub ---
    class _Settings:
        inference_prompt_registry_path = "prompts/prompt_registry.yaml"
        prompt_domain_default = ""
        summarizer_model = "gpt-4o-mini"
        inference_model = "gpt-4o-mini"
        summarizer_temperature = 0.0
        summarizer_max_output_tokens = 64

    settings = _Settings()

    mgr = chm.ChunkedHistoryManager(chunk_size_limit=2, session_id="test")

    # Build a 2-turn history (4 messages) so rollover condition is true on the next call.
    history = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]

    # Simulate two completed turns.
    mgr.increment_turn_count()
    mgr.increment_turn_count()

    assert mgr.should_create_new_chunk() is True, "Expected rollover after 2 turns"

    ok = mgr.create_new_chunk(history, settings, cache={}, namespace="ns-test")
    if not ok:
        print("FAIL: create_new_chunk returned False")
        return 1

    # Verify state reset + summary updated
    recent, summary = mgr.get_history_for_prompt(history)
    print("ok=", ok)
    print("accumulated_summary=", summary)
    print("current_chunk_start=", mgr.current_chunk_start)
    print("current_chunk_size=", mgr.current_chunk_size)
    print("recent_len=", len(recent))

    if not summary:
        print("FAIL: accumulated_summary is empty")
        return 1

    if mgr.current_chunk_size != 0:
        print("FAIL: current_chunk_size should reset to 0")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
