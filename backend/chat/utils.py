"""Utility functions for chat processing.

Shared utilities that can be used across different chat modules.
"""

from typing import Dict, Any, List, Tuple


def _get_param_int(params: Dict[str, Any] | None, keys: List[str], default: int, minimum: int | None = None, maximum: int | None = None) -> tuple[int, str]:
    """
    Return (value, source) reading the first available key in `keys` from params, else the `default`.
    Coerces to int and clamps to [minimum, maximum] if provided.
    """
    if params:
        for key in keys:
            if key in params:
                try:
                    value = int(params[key])
                    if minimum is not None:
                        value = max(minimum, value)
                    if maximum is not None:
                        value = min(maximum, value)
                    return value, key
                except (ValueError, TypeError):
                    # Invalid int, continue to next key or default
                    continue
    return default, f"settings:{keys[0]}"


def split_history_for_prompt(history_msgs: List[Dict[str, str]] | None, raw_tail_turns: int, window_turns: int):
    """
    Split a flat message list into (to_summarize, verbatim_tail).

    Definitions:
      - 1 turn = 2 messages (user + assistant).
      - verbatim_tail: the last `raw_tail_turns` COMPLETED turns (up to available), kept verbatim in the prompt.
      - to_summarize: `window_turns` turns immediately before the tail, used to build a short summary.
    
    IMPORTANT: Only excludes the current ongoing turn (user question + processing message) from both
    the summary and tail to ensure only completed conversation turns are included.
    Does NOT exclude completed turns.

    This function is pure and does not modify the input list.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    msgs = history_msgs or []
    msgs_per_turn = 2
    
    # Check if the last 2 messages represent a current turn (user + processing)
    # Only exclude them if they match this pattern
    completed_msgs = msgs[:]
    if len(msgs) >= 2:
        last_msg = msgs[-1]
        second_last_msg = msgs[-2]
        
        # Check if this looks like a current turn:
        # - Last message is assistant with "processing" content
        # - Second to last is user
        is_processing = (
            str(last_msg.get('role', '')).strip() == 'assistant' and
            str(last_msg.get('content', '')).strip().lower() == 'processing'
        )
        is_user_question = str(second_last_msg.get('role', '')).strip() == 'user'
        
        if is_processing and is_user_question:
            # This looks like a current turn, exclude it
            completed_msgs = msgs[:-2]
        else:
            # This doesn't look like a current turn, keep all messages
            pass
    
    tail_msg_count = max(0, int(raw_tail_turns)) * msgs_per_turn
    window_msg_count = max(0, int(window_turns)) * msgs_per_turn

    total = len(completed_msgs)
    # Verbatim tail = last K COMPLETED turns (2*K messages)
    verbatim_tail = completed_msgs[-tail_msg_count:] if tail_msg_count > 0 else []

    # Summary window = the turns immediately before the tail (2*window_turns messages)
    end = max(0, total - tail_msg_count)
    start = max(0, end - window_msg_count)
    to_summarize = completed_msgs[start:end]

    return to_summarize, verbatim_tail
