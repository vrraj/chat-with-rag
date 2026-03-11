"""
Chunked History Manager for better cache efficiency.

Manages conversation history in chunks to improve cache hit rates and token efficiency.
Instead of sliding window (which creates new cache keys every turn), uses fixed chunks
with accumulated summaries.
"""

import logging
from typing import List, Dict, Any, Optional
from backend.llm.llm_client import generate, LLMError

logger = logging.getLogger(__name__)


class ChunkedHistoryManager:
    """
    Manages conversation history in chunks for better cache efficiency.
    
    Instead of sliding window (cache miss every turn), uses:
    - Fixed-size chunks (e.g., 10 turns per chunk)
    - Accumulated summary of all previous chunks
    - Current chunk sent verbatim
    """
    
    def __init__(self, chunk_size_limit: int = 10, session_id: str = "default"):
        self.chunk_size_limit = chunk_size_limit
        self.session_id = session_id
        
        # Track current chunk state
        self.current_chunk_start = 0      # History index where current chunk started
        self.current_chunk_size = 0       # Number of turns in current chunk
        
        # Accumulated summary of all previous chunks
        self.accumulated_summary = ""
        
        # Track if we've started chunking (first chunk creation)
        self.has_started_chunking = False

        # If summary_update is misconfigured, repeated rollover attempts are expensive.
        # Disable further rollover attempts for this manager until it is reset/cleared.
        self.disable_rollover: bool = False
        
        logger.info(f"[CHUNKED] Initialized chunk manager for session {session_id}, chunk_size={chunk_size_limit}")
    
    def should_create_new_chunk(self) -> bool:
        """Check if we've reached the chunk size limit and need to create a new chunk."""
        return self.current_chunk_size >= self.chunk_size_limit
    
    def should_create_new_chunk_by_tokens(self, messages: List[Dict[str, str]], token_limit: int) -> bool:
        """Check if we've reached the token limit for token-based chunks."""
        if not messages:
            return False
        
        # Simple token counting (in future, use actual tokenizer)
        total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
        estimated_tokens = total_chars // 4  # Rough estimate: 1 token ≈ 4 chars
        
        return estimated_tokens >= token_limit
    
    def get_current_chunk_messages(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Get messages from the current chunk only.
        
        Args:
            history: Full conversation history
            
        Returns:
            Messages from current chunk start to end, excluding trailing "Processing" placeholder
        """
        if not history:
            return []
        
        # Convert turn-based index to message-based index (2 messages per turn)
        chunk_start_msg_idx = self.current_chunk_start * 2
        recent = history[chunk_start_msg_idx:]
        
        # Exclude trailing "Processing" placeholder if present
        if recent and recent[-1].get("role") == "assistant" and recent[-1].get("content", "").strip() == "Processing":
            recent = recent[:-1]
        
        return recent
    
    def create_new_chunk(
        self, 
        history: List[Dict[str, str]], 
        settings_obj: Any,
        cache: Dict[str, Any],
        namespace: Optional[str] = None
    ) -> bool:
        """
        Create a new chunk by summarizing the current chunk and resetting.
        
        Args:
            history: Full conversation history
            settings_obj: Settings object for model configuration
            cache: Cache for summary storage
            namespace: Optional namespace for cache keys
            
        Returns:
            True if chunk was created successfully, False otherwise
        """
        try:
            if self.disable_rollover:
                logger.warning(f"[CHUNKED] Rollover disabled for session {self.session_id}; skipping chunk creation")
                return False
            # Get current chunk messages
            current_chunk = self.get_current_chunk_messages(history)
            
            if not current_chunk:
                logger.warning(f"[CHUNKED] No current chunk messages to summarize for session {self.session_id}")
                return False
            
            logger.info(f"[CHUNKED] Creating new chunk for session {self.session_id}, chunk has {len(current_chunk)} messages")
            
            # Update summary with current chunk
            success = self._update_summary(current_chunk, settings_obj, cache, namespace)
            
            if success:
                # Reset for new chunk
                self._reset_for_new_chunk(history)
                logger.info(f"[CHUNKED] Successfully created new chunk for session {self.session_id}")
                return True
            else:
                logger.error(f"[CHUNKED] Failed to update summary for session {self.session_id}")
                try:
                    self.disable_rollover = True
                except Exception:
                    pass
                return False
                
        except Exception as e:
            logger.error(f"[CHUNKED] Error creating new chunk for session {self.session_id}: {e}")
            return False
    
    def _update_summary(
        self, 
        new_conversation: List[Dict[str, str]], 
        settings_obj: Any,
        cache: Dict[str, Any],
        namespace: Optional[str] = None
    ) -> bool:
        """
        Update accumulated summary with new conversation chunk.
        
        Uses the summary_update prompt to intelligently merge new conversation
        with existing summary.
        """
        try:
            from backend.chat.prompt_registry import resolve_summary_update_prompt, render_full_payload
            
            # Format conversation for prompt
            conversation_text = self._format_conversation_for_prompt(new_conversation)
            
            # Resolve the summary update prompt
            registry_path = getattr(settings_obj, "inference_prompt_registry_path", "")
            prompt_domain = getattr(settings_obj, "prompt_domain_default", "")
            
            sum_spec = resolve_summary_update_prompt(registry_path=registry_path, domain=prompt_domain)
            
            # Render payload with template variables
            payload = render_full_payload(
                sum_spec.full_payload_template,
                variables={
                    "prior_chat_summary": self.accumulated_summary,
                    "recent_conversation": conversation_text,
                }
            )
            
            # Call LLM to update summary
            model = getattr(settings_obj, "summarizer_model", settings_obj.inference_model)
            temperature = float(getattr(settings_obj, "summarizer_temperature", 0.3))
            max_output_tokens = int(getattr(settings_obj, "summarizer_max_output_tokens", 128))
            
            logger.debug(f"[CHUNKED] Updating summary with {len(new_conversation)} messages for session {self.session_id}")

            # Use the generate function for consistent normalized response
            resp = generate(
                model_key=model,
                input=[
                    {"role": "system", "content": sum_spec.system_instruction},
                    {"role": "user", "content": payload},
                ],
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

            updated_summary = ""
            try:
                # generate() returns normalized response
                updated_summary = str(resp.get("text") or "").strip()
            except Exception:
                updated_summary = ""
            
            if updated_summary:
                self.accumulated_summary = updated_summary
                logger.info(f"[CHUNKED] Updated summary for session {self.session_id}: {updated_summary}")
                return True
            else:
                logger.warning(f"[CHUNKED] Empty summary response for session {self.session_id}")
                return False
                
        except LLMError as e:
            logger.error(f"[CHUNKED] LLM error updating summary for session {self.session_id}: {e}")
            try:
                if "Prompt registry missing required" in str(e):
                    self.disable_rollover = True
            except Exception:
                pass
            return False
        except Exception as e:
            logger.error(f"[CHUNKED] Unexpected error updating summary for session {self.session_id}: {e}")
            return False
    
    def _reset_for_new_chunk(self, history: List[Dict[str, str]]) -> None:
        """Reset chunk tracking for the next chunk."""
        # Set new chunk start to the last completed turn so the next turn's recent_conversation includes it
        self.current_chunk_start = max(0, (len(history) // 2) - 1)
        self.current_chunk_size = 0
        self.has_started_chunking = True
        
        logger.debug(f"[CHUNKED] Reset chunk tracking for session {self.session_id}, new start at turn {self.current_chunk_start}")
    
    def _format_conversation_for_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation messages for the summary prompt."""
        lines = []
        for msg in messages:
            role = msg.get("role", "user").strip()
            content = msg.get("content", "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)
    
    def get_history_for_prompt(
        self, 
        history: List[Dict[str, str]]
    ) -> tuple[List[Dict[str, str]], str]:
        """
        Get history formatted for prompt construction.
        
        Returns:
            Tuple of (recent_conversation, accumulated_summary)
        """
        recent_conversation = self.get_current_chunk_messages(history)
        summary = self.accumulated_summary
        
        logger.debug(f"[CHUNKED] Returning {len(recent_conversation)} recent messages, summary length: {len(summary)}")
        
        return recent_conversation, summary
    
    def increment_turn_count(self) -> None:
        """Increment the turn count for the current chunk."""
        self.current_chunk_size += 1
        logger.debug(f"[CHUNKED] Incremented turn count to {self.current_chunk_size}/{self.chunk_size_limit} for session {self.session_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the chunk manager."""
        return {
            "session_id": self.session_id,
            "chunk_size_limit": self.chunk_size_limit,
            "current_chunk_size": self.current_chunk_size,
            "current_chunk_start": self.current_chunk_start,
            "accumulated_summary_length": len(self.accumulated_summary),
            "has_started_chunking": self.has_started_chunking,
            "needs_new_chunk": self.should_create_new_chunk()
        }
