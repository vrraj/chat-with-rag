from typing import List, Dict, Any, Optional
import openai
import uuid
from backend.core.config import settings
from backend.embeddings.qdrant_client import QdrantStorage
from backend.embeddings.text_splitter import TextSplitter
from qdrant_client import models
from backend.embeddings.collection_manager import CollectionManager

class EmbeddingsManager:
    def __init__(self):
        self.qdrant: QdrantStorage = QdrantStorage()
        self.collection_manager = CollectionManager(self.qdrant.client)
        self.client = openai.OpenAI(api_key=settings.openai_api_key)

    def get_chunks_by_url(self, url: str, limit: int = 100) -> List[Dict]:
        """
        Retrieve chunks from Qdrant that match the given URL in the payload.
        """
        try:
            print(f"[DEBUG] Retrieving chunks for URL: {url}")
            points, _ = self.qdrant.client.scroll(
                collection_name=self.qdrant.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="url",
                            match=models.MatchValue(value=url)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            print(f"[DEBUG] Found {len(points)} matching chunks for URL: {url}")
            return [
                {
                    "id": point.id,
                    "payload": point.payload
                }
                for point in points
            ]
        except Exception as e:
            print(f"[ERROR] Failed to scroll Qdrant for URL {url}: {e}")
            return []

    def generate_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings using OpenAI's API
        Args:
            text: Text to embed
        Returns:
            List of float values representing the embedding
        """
        try:
            #print(f"[DEBUG] Generating embedding using model: {settings.embedding_model}")
            response = self.client.embeddings.create(
                input=text,
                model=settings.embedding_model
            )
            #print(f"[DEBUG] Raw OpenAI response data: {response.data}")
            embedding = response.data[0].embedding
            #print(f"[DEBUG] Type of embedding: {type(embedding)}, first 5 elements: {embedding[:5]}")
            #print(f"[DEBUG] Embedding as Python list: {list(embedding)}")
            prompt_tokens = response.usage.prompt_tokens if response.usage else "N/A"
            total_tokens = response.usage.total_tokens if response.usage else "N/A"
            #print(f"[DEBUG] Tokens used - prompt: {prompt_tokens}, total: {total_tokens}")
            #print(f"[DEBUG] Embedding generated. Input tokens used: {total_tokens}")
            #print(f"[DEBUG] Received embedding vector of length: {len(embedding)}")
            return embedding
        except Exception as e:
            print(f"Error generating embeddings: {e}")
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
        # Ensure collection exists before indexing
        try:
            self.qdrant.client.get_collection(settings.collection_name)
        except Exception:
            print(f"[DEBUG] Collection not found, creating it now...")
            self.qdrant.create_collection()

        url = document.get('url')
        if url and force_delete:
            print(f"[DEBUG] Force delete is enabled. Deleting existing entries for: {url}")
            self.delete_document(url)
        # Process document and generate embeddings  
        #print(f"[DEBUG] Processing chunks for document: {url}")
        processed_chunks = self.process_document(document)
        try:
            #print(f"[DEBUG] Inserting {len(processed_chunks)} vectors into Qdrant collection: {self.qdrant.collection_name}")
            success = self.qdrant.add_embeddings(processed_chunks)
            if not success:
                raise Exception("Failed to add embeddings to Qdrant")
            print(f"[DEBUG] Successfully added {len(processed_chunks)} embeddings to Qdrant")
        except Exception as e:
            print(f"Error indexing embeddings to Qdrant: {e}")

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
        query_embedding = self.generate_embeddings(query)
        qdrant_filter = None
        if query_filter:
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="payload.url",
                        match=models.MatchValue(value=query_filter["url"])
                    )
                ]
            )
        
        results = self.qdrant.search(query_embedding, limit=limit, filter=qdrant_filter)
        
        return [{
            "score": result.score,
            "payload": result.payload
        } for result in results]

    def delete_document(self, url: str):
        """
        Delete all embeddings associated with a document
        Args:
            url: URL of the document to delete
        """
        try:
            # Debug: log attempt to scroll
            print(f"[DEBUG] Attempting to scroll for URL: {url}")
            # Get all points with this URL
            points = self.qdrant.client.scroll(
                collection_name=self.qdrant.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="url",
                            match=models.MatchValue(value=url)
                        )
                    ]
                )
            )
            # Debug: log raw scroll result length
            print(f"[DEBUG] Scroll returned {len(points[0])} points for URL: {url}")
            # Extract IDs and delete
            ids = [point.id for point in points[0]]  # points is a tuple (points, next_token)
            if ids:
                print(f"[DEBUG] Deleting {len(ids)} points for URL: {url}")
                self.qdrant.client.delete(
                    collection_name=self.qdrant.collection_name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="url",
                                    match=models.MatchValue(value=url)
                                )
                            ]
                        )
                    )
                )
                print(f"[DEBUG] Successfully deleted points for URL: {url}")
            else:
                print(f"[DEBUG] No points found to delete for URL: {url}")
        except Exception as e:
            print(f"[ERROR] Failed to delete document: {e}")
            raise

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