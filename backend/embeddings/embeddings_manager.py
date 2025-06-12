from typing import List, Dict, Any, Optional
import openai
import uuid
import logging
from backend.core.config import settings
from backend.embeddings.qdrant_client import QdrantStorage
from backend.embeddings.text_splitter import TextSplitter
from qdrant_client import models
from backend.embeddings.collection_manager import CollectionManager
from backend.db.qdrant_db import QdrantDB

logger = logging.getLogger(__name__)

class EmbeddingsManager:
    def __init__(self):
        self.qdrant: QdrantStorage = QdrantStorage()
        self.collection_manager = CollectionManager(self.qdrant.client)
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.qdrant_db = QdrantDB(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.collection_name
        )

    def generate_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings using OpenAI's API
        Args:
            text: Text to embed
        Returns:
            List of float values representing the embedding
        """
        try:
            logger.debug(f"Generating embedding using model: {settings.embedding_model}")
            response = self.client.embeddings.create(
                input=text,
                model=settings.embedding_model
            )
            embedding = response.data[0].embedding
            prompt_tokens = response.usage.prompt_tokens if response.usage else "N/A"
            total_tokens = response.usage.total_tokens if response.usage else "N/A"
            logger.debug(f"Tokens used - prompt: {prompt_tokens}, total: {total_tokens}")
            logger.debug(f"Received embedding vector of length: {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def process_document(self, document: Dict) -> List[Dict]:
        """
        Process a document by splitting it into chunks and generating embeddings
        Args:
            document: Dictionary containing text and metadata
        Returns:
            List of processed chunks with embeddings
        """
        max_chars = document.get("max_chars", None)
        print(f"[DEBUG] max_chars received from document: {max_chars}")
        text = document.get("text", "")
        if max_chars is not None:
            print(f"[DEBUG] Limiting input text to {max_chars} characters (original length: {len(text)})")
            truncated_text = text[:max_chars]
            print(f"[DEBUG] Final text length after truncation: {len(truncated_text)}")
        else:
            print(f"[DEBUG] No max_chars limit specified (original length: {len(text)})")
            truncated_text = text

        doc_type = document.get("doc_type", "HTML")
        if doc_type == "HTML":
            chunk_size = settings.html_chunk_size
            chunk_overlap = settings.html_chunk_overlap
        elif doc_type == "PDF":
            chunk_size = settings.pdf_chunk_size or len(text)
            chunk_overlap = settings.pdf_chunk_overlap
        else:
            doc_type = "text"
            chunk_size = settings.default_chunk_size
            chunk_overlap = settings.default_chunk_overlap

        #print(f"[DEBUG] Document type detected: {doc_type}")
        print(f"[DEBUG] Chunk Size: {chunk_size}")
        print(f"[DEBUG] Chunk Overlap: {chunk_overlap}")

        # Use Langchain-based token splitter (not manual tiktoken-based splitter)
        text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_manual_splitter=False
        )
        print(f"[DEBUG] Using chunk size: {text_splitter.chunk_size}")
        print(f"[DEBUG] Using chunk overlap: {text_splitter.chunk_overlap}")
        # Only pass the raw text to split_text
        chunks = text_splitter.split_text(truncated_text)
        #print(f"[DEBUG] Number of chunks generated: {len(chunks)}")
        # Attach MediaWiki-style metadata manually
        processed_chunks = []
        total_chunks = len(chunks)
        for idx, chunk_text in enumerate(chunks):
            try:
                chunk_id = str(uuid.uuid4())
                print(f"[DEBUG] Embedding chunk ID: {chunk_id}")
                embedding = self.generate_embeddings(chunk_text)
                processed_chunks.append({
                    "id": chunk_id,
                    "vector": embedding,
                    "payload": {
                        "text": chunk_text,
                        "section": None,
                        "subsection": None,
                        "chunk_index": idx,
                        "total_chunks": total_chunks,
                        "url": document.get("url", ""),
                        "document_type": document.get("document_type", "HTML"),
                        "source": document.get("url", ""),
                        "section_index": None,
                        "subsection_index": None,
                        "title": document.get("title", ""),
                        "description": document.get("description", ""),
                    }
                })
            except Exception as e:
                print(f"Error processing chunk {chunk_id}: {e}")
                continue

        return processed_chunks

    def index_document(self, document: Dict, force_delete: bool = True):
        """
        Index a document by processing it and storing embeddings
        
        Args:
            document: Dictionary containing text and metadata
            force_delete: If True, deletes existing Qdrant entries for the document URL before indexing
        """
        try:
            logger.debug(f"Indexing document: {document.get('url', 'unknown')}")
            # Ensure collection exists before indexing
            try:
                self.qdrant.client.get_collection(settings.collection_name)
            except Exception:
                logger.debug("Collection not found, creating it now...")
                self.qdrant.create_collection()

            url = document.get('url')
            if url and force_delete:
                logger.debug(f"Force delete is enabled. Deleting existing entries for: {url}")
                self.delete_document(url)
            
            # Process document and generate embeddings
            processed_chunks = self.process_document(document)
            
            try:
                logger.debug(f"Inserting {len(processed_chunks)} vectors into Qdrant collection: {self.qdrant.collection_name}")
                success = self.qdrant.add_embeddings(processed_chunks)
                if not success:
                    raise Exception("Failed to add embeddings to Qdrant")
                logger.debug(f"Successfully added {len(processed_chunks)} embeddings to Qdrant")
            except Exception as e:
                logger.error(f"Error indexing embeddings to Qdrant: {e}")
                raise
        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            raise

    def search_similar(self, query: str, limit: int = 5, query_filter: Optional[Any] = None) -> List[Dict]:
        """
        Search for similar content using a query
        
        Args:
            query: Search query
            limit: Number of results to return
            filter: Optional Qdrant filter to narrow the search (e.g., by URL)
        
        Returns:
            List of search results with scores and content
        """
        try:
            logger.debug(f"Searching for query: {query}")
            query_embedding = self.generate_embeddings(query)
            qdrant_filter = None
            if query_filter:
                url = query_filter["url"]
                url_lower = url.lower()
                qdrant_filter = models.Filter(
                    should=[
                        models.FieldCondition(
                            key="url",
                            match=models.MatchValue(value=url)
                        ),
                        models.FieldCondition(
                            key="url_lower",
                            match=models.MatchValue(value=url_lower)
                        )
                    ]
                )
            
            results = self.qdrant.search(query_embedding, limit=limit, filter=qdrant_filter)
            logger.debug(f"Found {len(results)} results for query: {query}")
            
            return [{
                "score": result.score,
                "payload": result.payload
            } for result in results]
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            raise

    def delete_document(self, url: str) -> int:
        """
        Delete all embeddings associated with a document
        
        Args:
            url: URL of the document to delete
            
        Returns:
            Number of points deleted
        """
        return self.qdrant_db.delete_by_url(url)

    def build_url_filter(self, url: str):
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="url",
                    match=models.MatchValue(value=url)
                )
            ]
        )

    def index_chunks(self, chunks: List[Dict], force_delete: bool = True):
        """
        Index pre-chunked data into Qdrant. Generates embeddings for each chunk and wraps in Qdrant format.
        Args:
            chunks: List of dicts, each with at least 'text' and metadata (not pre-embedded)
            force_delete: If True, deletes existing Qdrant entries for the URL(s) before indexing
        """
        if not chunks:
            print("[DEBUG] No chunks to index.")
            return

        print(f"[DEBUG] Indexing {len(chunks)} pre-chunked entries")

        try:
            self.qdrant.client.get_collection(settings.collection_name)
        except Exception:
            print(f"[DEBUG] Collection not found, creating it now...")
            self.qdrant.create_collection()

        # Generate embeddings and wrap chunks
        points = []
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            if not text:
                print(f"[WARNING] Skipping empty chunk at index {i}")
                continue

            embedding = self.generate_embeddings(text)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk.get('url', '')}-{i}"))
            points.append({
                "id": point_id,
                "vector": embedding,
                "payload": chunk
            })

        if force_delete:
            url_set = {point["payload"].get("url") for point in points}
            for url in url_set:
                if url:
                    #print(f"[DEBUG] Force delete is enabled. Deleting existing entries for: {url}")
                    self.delete_document(url)

        # Debug print payload URL for each point before adding embeddings (commented out)
        # for p in points:
        #     print(f"[DEBUG] Payload URL for point {p['id']}: {p['payload'].get('url')}")
        success = self.qdrant.add_embeddings(points)
        if not success:
            raise Exception("Failed to add embeddings to Qdrant")
        #print(f"[DEBUG] Successfully added {len(points)} pre-chunked embeddings to Qdrant")