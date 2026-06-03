"""Query rewrite helper functions."""

import logging
import re

logger = logging.getLogger(__name__)

# Query rewrite helpers
_REWRITE_DEICTIC_RE = re.compile(r"\b(it|this|that|these|those|here|there|they|them|their|its|he|she|his|her|also|then)\b", re.I)
_REWRITE_SHORT_Q_RE = re.compile(
    r"\b(what about|how about|how far|how long|how much|where is|when is|which one)\b",
    re.I,
)


def should_rewrite(message: str) -> bool:
    """Heuristic: return True if the message is likely underspecified (coreference or very short).
    Safe default: if this returns False, we skip rewrite and use the original message.
    Diagnostic logging included.
    """
    if not message:
        logger.debug("[REWRITE] heuristic=empty_message -> False")
        return False
    txt = message.strip()
    # Rewrite deictic follow-ups like:
    # "how far is it", "what about that one", "where is it"
    if _REWRITE_DEICTIC_RE.search(txt):
        logger.debug("[REWRITE] heuristic=deictic -> True")
        return True

    # Short follow-up style questions without explicit entities.
    # Avoid rewriting generic conversational/support requests like:
    # "can I talk to someone"
    if len(txt.split()) <= 7 and _REWRITE_SHORT_Q_RE.search(txt):
        logger.debug(
            "[REWRITE] heuristic=short_followup_question words=%d -> True",
            len(txt.split()),
        )
        return True
    logger.debug("[REWRITE] heuristic=none -> False")
    return False
