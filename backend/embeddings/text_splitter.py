from typing import List, Dict
import re
from langchain.text_splitter import TokenTextSplitter

"""Splits text into token-based chunks using TokenTextSplitter."""

class TextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = TokenTextSplitter(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split text into token-based chunks with metadata
        Args:
            text: The text to split
            metadata: Additional metadata to attach to each chunk
        Returns:
            List of chunks with metadata
        """
        chunks = self.splitter.split_text(text)
        
        print(f"[DEBUG] Total chunks created: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"[DEBUG] Chunk {i}: {chunk[:100]}...")
        print()  # Add a blank line for better readability

        # Create unique IDs for each chunk
        base_id = f"{metadata.get('url', '')}-{hash(text)}"

        # Avoid including original full document text in each chunk's payload
        # Only retain other metadata like title, domain, headers, etc.
        clean_metadata = {k: v for k, v in metadata.items() if k != "text"}

        # Return a list of structured chunks with consistent fields
        return [{
            "id": f"{base_id}-{i}",       # Unique chunk ID using base + index
            "text": chunk,                # The actual chunk content to embed
            **clean_metadata,             # All other metadata except original 'text'
            "chunk_index": i,             # Position of this chunk
            "total_chunks": len(chunks)   # Total number of chunks generated
        } for i, chunk in enumerate(chunks)]

    def split_document(self, document: Dict) -> List[Dict]:
        """
        Split a document with content and metadata
        Args:
            document: Dictionary containing text and metadata
        Returns:
            List of chunks with metadata
        """
        print(f"[DEBUG] Document received in split_document: {document}")
        print(f"[DEBUG] Calling split_text with document content length: {len(document.get('content', ''))}")
        return self.split_text(
            document.get("text", ""),
            {
                "title": document.get("title", ""),
                "url": document.get("url", ""),
                "headers": document.get("headers", []),
                "date": document.get("date", ""),
                "document_type": document.get("document_type", "HTML"),
                "domain": document.get("domain", "")
            }
        )
