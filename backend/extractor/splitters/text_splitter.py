from typing import List, Dict, Optional, Any
import re
import tiktoken
from langchain.text_splitter import TokenTextSplitter

import logging
from backend.core.config import settings

logger = logging.getLogger(__name__)

"""
A flexible text splitter that can use either Langchain's TokenTextSplitter or manual tiktoken-based splitting.

This class provides a unified interface for splitting text into token-based chunks, with the ability to
choose between using Langchain's built-in TokenTextSplitter or a custom manual implementation using tiktoken.
"""

class TextSplitter:
    """
    Initialize a text splitter that can use either Langchain's TokenTextSplitter or manual tiktoken-based splitting.
    
    Args:
        chunk_size: Maximum number of tokens per chunk
        chunk_overlap: Number of overlapping tokens between chunks
        use_manual_splitter: If True, uses manual tiktoken-based splitting instead of Langchain's TokenTextSplitter.
                             This is useful when you need more control over the splitting process or want to avoid
                             Langchain dependencies.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, use_manual_splitter: bool = False):
        # Ensure progress in manual splitting: overlap must be < chunk_size
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap >= chunk_size:
            # Adjust to a safe value to avoid non-advancing windows
            chunk_overlap = max(0, chunk_size - 1)
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.use_manual_splitter: bool = use_manual_splitter
        # If not using manual splitter, initialize Langchain's TokenTextSplitter
        if not use_manual_splitter:
            self.splitter: TokenTextSplitter = TokenTextSplitter(
                encoding_name="cl100k_base",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

    """
    Split text into token-based chunks.
    
    This method splits the input text into chunks of approximately `chunk_size` tokens, with overlapping
    chunks of size `chunk_overlap`. The method returns plain text chunks without any additional metadata.
    
    Args:
        text: The text to split into chunks
    
    Returns:
        List of strings, each containing a chunk of text
    """
    def split_text(self, text: str) -> List[str]:
        if self.use_manual_splitter:
            # Manual token-based splitting using tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text)
            #logger.debug("TextSplitter: manual mode token count=%d", len(tokens))
            chunks = []
            i = 0
            while i < len(tokens):
                chunk_ids = tokens[i:i + self.chunk_size]
                chunk_text = enc.decode(chunk_ids)
                chunks.append(chunk_text)
                if len(chunk_ids) == 0:
                    break
                i += self.chunk_size - self.chunk_overlap
            #logger.debug("TextSplitter: total chunks (manual)=%d", len(chunks))
        else:
            # Use Langchain's TokenTextSplitter for splitting
            chunks = self.splitter.split_text(text)
            enc = tiktoken.get_encoding("cl100k_base")
            token_count = len(enc.encode(text))
            #logger.debug("TextSplitter: langchain mode token count=%d", token_count)
            #logger.debug("TextSplitter: total chunks (langchain)=%d", len(chunks))

        if getattr(settings, "debug_verbose", False):
            maxc = int(getattr(settings, "debug_log_truncate_chars", 500))
            for i, chunk in enumerate(chunks[:3]):
                preview = chunk[: min(100, maxc)]
                #logger.debug("TextSplitter: chunk %d preview: %s...", i, preview)
        
        return chunks

