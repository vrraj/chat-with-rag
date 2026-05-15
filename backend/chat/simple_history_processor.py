"""Simple History Processor Module

Focused helper class for consistent recent conversation formatting only.
Extracted from chat_manager.py to ensure byte-level consistency.
"""

from typing import List, Dict, Any, Optional
import logging
import hashlib
import re

logger = logging.getLogger(__name__)


class SimpleHistoryProcessor:
    """Handles consistent formatting of recent conversation for prompt injection."""
    
    def __init__(self, settings_obj: Any):
        """Initialize the history processor with settings object."""
        self.settings_obj = settings_obj
    
    def format_recent_conversation(
        self, 
        verbatim_tail: List[Dict[str, Any]], 
        params: Optional[Dict[str, Any]], 
        log_origin: str = "simple_processor"
    ) -> str:
        """
        Format recent conversation messages for prompt injection.
        
        Ensures byte-level consistency by:
        1. Deterministic role resolution
        2. Consistent string formatting
        3. Predictable ordering
        4. Stable content cleaning
        """
        
        # Create a deterministic copy to avoid modifying the original
        _tail = [dict(msg) for msg in verbatim_tail]
        
        # Get assistant role once to ensure consistency
        assistant_role = self._get_assistant_role(params)
        
        # Remove processing message from tail (deterministic)
        original_tail_length = len(_tail)
        
        
        # Remove processing message(s) from the end
        while _tail:
            last_msg = _tail[-1]
            role = str(last_msg.get("role", "")).strip()
            content = str(last_msg.get("content", "")).strip().lower()
            
            # Check for processing message in both original and target role formats
            # Handle various forms: "processing", "Processing", " processing ", etc.
            is_assistant = (role == "assistant" or role == assistant_role)
            is_processing = (content == "processing")
            
            
            if is_assistant and is_processing:
                _tail.pop()
            else:
                # No more processing messages at the end
                break
        
        
        # Format each message deterministically
        tail_lines: List[str] = []
        trimmed = 0
        
        for msg in _tail:
            # Normalize role and content strings consistently
            role = str(msg.get("role", "user")).strip()
            content = str(msg.get("content", "")).strip()
            
            
            # Clean sources from both original and target role formats
            if role == "assistant" or role == assistant_role:
                cleaned = self._strip_sources_deterministic(content)
                if cleaned != content:
                    trimmed += 1
                content = cleaned
                # Convert to target role if it was the original "assistant" role
                if role == "assistant":
                    role = assistant_role
            
            # Format line consistently (no extra spaces, predictable structure)
            tail_lines.append(f"{role}: {content}")
        
        # Join with consistent line endings
        recent_block_str = "\n".join(tail_lines) + "\n\n"
        
        
        if trimmed:
            pass
        
        return recent_block_str
    
    def _get_assistant_role(self, params: Optional[Dict[str, Any]]) -> str:
        """Get assistant role in a deterministic way."""
        try:
            from backend.core.config import get_assistant_role
            role = get_assistant_role(self.settings_obj, params)
            if role:
                result = str(role).strip()
                return result
        except Exception as e:
            pass
        
        # Fallback to default
        default_role = str(getattr(self.settings_obj, 'assistant_role_default', 'assistant') or 'assistant').strip()
        return default_role
    
    def _strip_sources_deterministic(self, content: str) -> str:
        """
        Strip sources block in a completely deterministic way.
        
        This ensures that the same input always produces the same output
        at the byte level, regardless of environmental factors.
        """
        try:
            # Normalize line endings first to ensure consistency
            normalized = content.replace('\r\n', '\n').replace('\r', '\n')
            
            # Strip trailing whitespace consistently
            s = normalized.rstrip()
            
            # Handle actual Sources format: content.sources:[1] https://... (same line)
            pattern = re.compile(r"(?:\n)?(?:<sources>Sources</sources>|Sources|sources):[\s\S]*\Z")
            m = pattern.search(s)
            if m:
                s = s[:m.start()]
            
            # Final strip with consistent behavior
            return s.rstrip()
        except Exception:
            # Return original content (normalized) if anything fails
            try:
                return content.replace('\r\n', '\n').replace('\r', '\n').rstrip()
            except Exception:
                return content or ""
    
    def verify_consistency(
        self, 
        verbatim_tail: List[Dict[str, Any]], 
        params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verify that the recent conversation formatting is consistent.
        
        Returns a dictionary with:
        - 'consistent': bool indicating if formatting is consistent
        - 'hash': SHA256 hash of the formatted output
        - 'formatted_output': the actual formatted output
        - 'error': any error message if consistency check failed
        """
        try:
            # Generate formatted output
            formatted_output = self.format_recent_conversation(
                verbatim_tail, params, "consistency_check"
            )
            
            # Generate hash for byte-level comparison
            hash_value = hashlib.sha256(formatted_output.encode('utf-8')).hexdigest()
            
            # Run the formatting multiple times to ensure deterministic behavior
            for i in range(3):
                test_output = self.format_recent_conversation(
                    verbatim_tail, params, "consistency_check"
                )
                test_hash = hashlib.sha256(test_output.encode('utf-8')).hexdigest()
                
                if test_hash != hash_value:
                    return {
                        'consistent': False,
                        'hash': hash_value,
                        'formatted_output': formatted_output,
                        'error': f'Inconsistent output detected on iteration {i+1}: {hash_value} vs {test_hash}'
                    }
            
            return {
                'consistent': True,
                'hash': hash_value,
                'formatted_output': formatted_output,
                'error': None
            }
            
        except Exception as e:
            return {
                'consistent': False,
                'hash': None,
                'formatted_output': None,
                'error': f'Consistency check failed: {str(e)}'
            }
    
    def get_consistency_hash(
        self, 
        verbatim_tail: List[Dict[str, Any]], 
        params: Optional[Dict[str, Any]]
    ) -> str:
        """
        Get a consistency hash for the given verbatim tail and params.
        
        This can be used to track if the same conversation history
        produces the same formatted output across different runs.
        """
        result = self.verify_consistency(verbatim_tail, params)
        return result['hash'] if result['consistent'] else 'inconsistent'
