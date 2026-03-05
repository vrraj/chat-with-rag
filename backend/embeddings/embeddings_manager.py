from typing import List, Dict, Any, Optional
import uuid
import logging
import time
from backend.core.config import settings
from backend.db.qdrant_client import QdrantStorage
from backend.extractor.splitters import TextSplitter
from qdrant_client import models
from backend.embeddings.collection_manager import CollectionManager
from backend.db import QdrantDB
from backend.embeddings.specs import resolve_embedding_spec
from backend.llm.llm_client import embed
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

def _strip_fragment(u: str) -> str:
    if not u:
        return u
    p = urlsplit(u)
    # keep scheme, host, path, query; drop fragment
    return urlunsplit((p.scheme, p.netloc, p.path, p.query, ""))

class EmbeddingsManager:
    def __init__(self):
        self.qdrant: QdrantStorage = QdrantStorage()
        self.collection_manager = CollectionManager(self.qdrant.client)
        # Initialize QdrantDB without the callback first
        self.qdrant_db = QdrantDB(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.collection_name,
        )
        # Now set the callback after the method is defined
        self.qdrant_db.generate_embeddings = self.generate_embeddings
        # Track tokens used during a single indexing operation
        self._tokens_used: int = 0
        

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in the text using tiktoken.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            int: Number of tokens in the text
        """
        try:
            import tiktoken
            # Use the same encoding as the embeddings model
            encoding = tiktoken.get_encoding("cl100k_base")  # cl100k_base is used by text-embedding-ada-002
            return len(encoding.encode(text, disallowed_special=()))
        except Exception as e:
            logger.warning(f"Failed to use tiktoken for token counting: {e}")
            # Fallback to character count estimation if tiktoken fails
            return max(1, len(text) // 4)

    def generate_embeddings(self, text: Any):
        """
        Generate embeddings for DOCUMENT INDEXING.

        Uses gemini_embed_type_documents config for optimal indexing performance.
        Backward-compatible behavior:
        - All embeddings now route through `embed()` regardless of model type.
        - The function automatically handles both legacy OpenAI model ids and provider keys.
        Args:
            text: Text to embed, or a list of texts for batched embeddings.
        Returns:
            - If `text` is a str: a single embedding vector (List[float]).
            - If `text` is a list of str: a list of embedding vectors (List[List[float]]).
        """
        attempt = 0
        backoff = max(0.0, float(settings.embeddings_initial_backoff_secs))
        last_err = None
        while attempt < int(settings.embeddings_max_retries):
            try:
                if settings.embeddings_call_delay_secs:
                    time.sleep(float(settings.embeddings_call_delay_secs))
                # Resolve the embedding spec so we can support both provider-mode
                # configs and legacy model-id configs.
                try:
                    spec = resolve_embedding_spec(settings)
                    provider = spec.get("provider", "openai")
                    model = spec.get("model")
                    dims = spec.get("dimensions")
                except Exception:
                    provider = "openai"
                    model = "text-embedding-3-small"
                    dims = 1536

                # Always use embed() for embeddings. Support both single-text
                # and batched (list-of-texts) inputs.
                is_batch = isinstance(text, list)
                kwargs: Dict[str, Any] = {
                    "provider": provider,
                    "model": model,
                    "input": text,
                }
                if provider == "gemini" and isinstance(dims, int) and dims > 0:
                    kwargs["dimensions"] = dims
                    # Apply config-driven task type for document indexing
                    try:
                        task_type = getattr(settings, "gemini_embed_type_documents", "RETRIEVAL_DOCUMENT")
                        kwargs["task_type"] = task_type
                        logger.debug(f"[GEMINI INDEXING] Using task_type={task_type} for document indexing")
                    except Exception:
                        pass
                    # Apply config-driven normalization flag so that Gemini
                    # embeddings are treated consistently for both indexing
                    # and query-time embeddings.
                    try:
                        kwargs["normalize_embedding"] = bool(settings.gemini_embedding_normalize)
                    except Exception:
                        pass

                # DEBUG: Log the resolved embedding provider/model/dimensions
                try:
                    logger.debug(
                        "[EMBEDDINGS] Generating embedding via provider=%s model=%s dimensions=%s",
                        provider,
                        model,
                        dims,
                    )
                except Exception:
                    pass

                # Remove provider from kwargs since it's inferred from model_key
                kwargs_for_embed = {k: v for k, v in kwargs.items() if k != "provider"}
                
                response = embed(model_key=model, texts=text, **kwargs_for_embed)
                # Normalize embeddings into a list for simpler handling.
                embeddings_list = [d.embedding for d in response.data]
                
                # Capture magnitude metadata if available
                magnitudes = []
                normalized_flags = []
                providers = []
                for d in response.data:
                    magnitudes.append(getattr(d, 'magnitude', None))
                    normalized_flags.append(getattr(d, 'normalized', False))
                    providers.append(getattr(d, 'provider', 'unknown'))
                
                prompt_tokens = response.usage.prompt_tokens if response.usage else "N/A"
                total_tokens = response.usage.total_tokens if response.usage else 0
                try:
                    self._tokens_used += int(total_tokens or 0)
                except Exception:
                    pass
                logger.debug("Tokens used - prompt: %s, total: %s", prompt_tokens, total_tokens)
                
                if is_batch:
                    # Return list-of-embeddings with metadata
                    return embeddings_list, magnitudes, normalized_flags, providers
                # Single-text path: return embedding with metadata
                return embeddings_list[0] if embeddings_list else [], (magnitudes[0] if magnitudes else None), (normalized_flags[0] if normalized_flags else False), (providers[0] if providers else 'unknown')
            except Exception as e:
                last_err = e
                attempt += 1
                logger.warning(
                    "Embedding call failed (attempt %s/%s): %s",
                    attempt, settings.embeddings_max_retries, e, exc_info=True
                )
                if attempt >= int(settings.embeddings_max_retries):
                    break
                # Exponential backoff
                sleep_for = max(0.0, backoff)
                time.sleep(sleep_for)
                backoff *= 2
        logger.error("Error generating embeddings after retries: %s", last_err)
        raise last_err

    def process_document(self, document: Dict, max_chunks: Optional[int] = None) -> List[Dict]:
        """
        Process a document by splitting it into chunks and generating embeddings
        Args:
            document: Dictionary containing text and metadata
        Returns:
            List of processed chunks with embeddings
        """
        text = document.get("text", "")

        # Handle both doc_type and document_type for backward compatibility
        doc_type = document.get("doc_type") or document.get("document_type", "HTML")
        normalized_doc_type = doc_type.lower()
        if normalized_doc_type == "html":
            chunk_size = settings.html_chunk_size
            chunk_overlap = settings.html_chunk_overlap
        elif normalized_doc_type == "pdf":
            chunk_size = settings.pdf_chunk_size or len(text)
            chunk_overlap = settings.pdf_chunk_overlap
        elif normalized_doc_type == "mediawiki":
            chunk_size = settings.mediawiki_chunk_size
            chunk_overlap = settings.mediawiki_chunk_overlap
        else:
            doc_type = "text"
            chunk_size = settings.default_chunk_size
            chunk_overlap = settings.default_chunk_overlap

        #logger.debug("Chunk size: %s", chunk_size)
        #logger.debug("Chunk overlap: %s", chunk_overlap)

        # Use Langchain-based token splitter (not manual tiktoken-based splitter)
        text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_manual_splitter=False
        )
        # Only pass the raw text to split_text
        chunks = text_splitter.split_text(text)
        # Safety cap to prevent runaway processing; respect user-provided max if supplied
        effective_cap = int(settings.max_chunks_per_doc)  # Hard limit set in config.py
        if max_chunks is not None:
            try:
                user_cap = int(max_chunks)
                if user_cap > 0:
                    effective_cap = min(effective_cap, user_cap)
            except Exception:
                pass
        if len(chunks) > effective_cap:
            #  logger.warning("Capping chunks from %d to %d", len(chunks), effective_cap)
            chunks = chunks[:effective_cap]
        # Attach MediaWiki-style metadata manually
        processed_chunks = []
        total_chunks = len(chunks)
        failures = 0
        started_at = time.time()

        # Get the original total_chunks from the first chunk if it exists
        original_total_chunks = None
        if chunks and hasattr(chunks[0], 'get') and 'total_chunks' in chunks[0]:
            original_total_chunks = chunks[0].get('total_chunks')

        # Resolve embedding spec once per document so we can record the model name in payloads
        try:
            _emb_spec_doc = resolve_embedding_spec(settings)
            _emb_model_name_doc = _emb_spec_doc.get("model")
            _emb_provider_doc = _emb_spec_doc.get("provider", "openai")
        except Exception:
            _emb_model_name_doc = "text-embedding-3-small"
            _emb_provider_doc = "openai"

        # Determine batch size based on embedding provider, with a safe default.
        try:
            if _emb_provider_doc == "openai":
                batch_size = int(getattr(settings, "embedding_batch_size_openai", 25))
            elif _emb_provider_doc == "gemini":
                batch_size = int(getattr(settings, "embedding_batch_size_gemini", 25))
            else:
                batch_size = int(getattr(settings, "embedding_batch_size_default", 25))
        except Exception:
            batch_size = 25
        if batch_size <= 0:
            batch_size = 1

        # DEBUG: Log the provider and chosen batch size for this document.
        try:
            logger.debug(
                "[EMBEDDINGS] Using provider=%s batch_size=%s for document chunks",
                _emb_provider_doc,
                batch_size,
            )
        except Exception:
            pass

        # Process chunks in batches, but keep token accounting and usage logic
        # centralized inside generate_embeddings.
        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start: batch_start + batch_size]
            try:
                logger.debug(
                    "[EMBEDDINGS] Embedding batch starting_at=%s batch_len=%s",
                    batch_start,
                    len(batch),
                )
            except Exception:
                pass
            batch_texts = []
            for chunk in batch:
                chunk_text = chunk if isinstance(chunk, str) else chunk.get('text', '')
                batch_texts.append(chunk_text)

            try:
                # generate_embeddings will handle provider routing, retries, and
                # token accounting. When passed a list, it should return a list
                # of embeddings aligned with batch_texts.
                if len(batch_texts) == 1:
                    embedding_result = self.generate_embeddings(batch_texts[0])
                    if isinstance(embedding_result, tuple):
                        # Single embedding with metadata
                        batch_embeddings = [embedding_result[0]]
                        batch_magnitudes = [embedding_result[1]]
                        batch_normalized = [embedding_result[2]]
                        batch_providers = [embedding_result[3]]
                    else:
                        # Legacy format (backward compatibility)
                        batch_embeddings = [embedding_result]
                        batch_magnitudes = [None]
                        batch_normalized = [False]
                        batch_providers = ['unknown']
                else:
                    embedding_result = self.generate_embeddings(batch_texts)  # type: ignore[arg-type]
                    if isinstance(embedding_result, tuple) and len(embedding_result) == 4:
                        # Batch embeddings with metadata
                        batch_embeddings, batch_magnitudes, batch_normalized, batch_providers = embedding_result
                    else:
                        # Legacy format (backward compatibility)
                        batch_embeddings = embedding_result
                        batch_magnitudes = [None] * len(batch_embeddings)
                        batch_normalized = [False] * len(batch_embeddings)
                        batch_providers = ['unknown'] * len(batch_embeddings)
            except Exception as e:
                failures += 1
                logger.error("Error generating embeddings for batch starting at %s: %s", batch_start, e, exc_info=True)
                if failures >= int(settings.embeddings_max_consecutive_failures_per_doc):
                    logger.error("Aborting document processing due to excessive embedding failures")
                    break
                # Time budget guard
                if (time.time() - started_at) > float(settings.embeddings_total_time_limit_secs):
                    logger.error("Aborting document processing due to time limit exceeded")
                    break
                continue

            for offset, chunk in enumerate(batch):
                idx = batch_start + offset
            try:
                chunk_id = str(uuid.uuid4())
                chunk_text = chunk if isinstance(chunk, str) else chunk.get('text', '')
                # Align embedding with chunk in the current batch.
                try:
                    embedding = batch_embeddings[offset]
                    magnitude = batch_magnitudes[offset] if offset < len(batch_magnitudes) else None
                    normalized = batch_normalized[offset] if offset < len(batch_normalized) else False
                    provider = batch_providers[offset] if offset < len(batch_providers) else 'unknown'
                except Exception:
                    # Fallback: regenerate singly if batch alignment fails for any reason.
                    embedding_result = self.generate_embeddings(chunk_text)
                    if isinstance(embedding_result, tuple):
                        embedding, magnitude, normalized, provider = embedding_result
                    else:
                        embedding, magnitude, normalized, provider = embedding_result, None, False, 'unknown'

                # Optional per-document token budget guard
                try:
                    max_tokens_budget = int(getattr(settings, "embeddings_max_tokens_per_doc", 0) or 0)
                except Exception:
                    max_tokens_budget = 0
                if max_tokens_budget > 0 and self._tokens_used > max_tokens_budget:
                    logger.error(
                        "Aborting document processing: embedding token budget exceeded (%s > %s)",
                        self._tokens_used,
                        max_tokens_budget,
                    )
                    break

                # Get URL from chunk metadata if available, otherwise from document
                url = chunk.get('url') if hasattr(chunk, 'get') else document.get("url", "")
                base = _strip_fragment(url)

                # Get chunk_index from chunk metadata if available, otherwise use loop index
                chunk_index = chunk.get('chunk_index', idx) if hasattr(chunk, 'get') else idx

                # Use original_total_chunks if available, otherwise use current batch size
                chunk_total_chunks = original_total_chunks if original_total_chunks is not None else total_chunks

                payload = {
                    "text": chunk_text,
                    "chunk_index": chunk_index,
                    "total_chunks": chunk_total_chunks,
                    "url": url,
                    "url_lower": url.lower(),  # Add lowercase version for case-insensitive filtering
                    "base_url": base,
                    "base_url_lower": (base or "").lower(),
                    "document_type": doc_type,
                    "source": url,
                    "title": document.get("title", ""),
                    "description": document.get("description", ""),
                }

                # Record the embedding model used for this vector (best-effort).
                if _emb_model_name_doc:
                    payload["embedding_model"] = _emb_model_name_doc
                
                # Add magnitude and normalization metadata if available
                if magnitude is not None:
                    payload["embedding_magnitude"] = magnitude
                payload["embedding_normalized"] = normalized
                payload["embedding_provider"] = provider

                # Prefer provided headings; default section to "Lead" if nothing present
                section = document.get("section") or document.get("section_title") or None
                if section is None:
                    section = "Lead"
                payload["section"] = section

                subsection = document.get("subsection") or document.get("subsection_title")
                if subsection:
                    payload["subsection"] = subsection
                else:
                    payload["subsection"] = None

                section_index = document.get("section_index")
                if section_index is not None:
                    payload["section_index"] = section_index
                else:
                    payload["section_index"] = None

                subsection_index = document.get("subsection_index")
                if subsection_index is not None:
                    payload["subsection_index"] = subsection_index
                else:
                    payload["subsection_index"] = None

                processed_chunks.append({
                    "id": chunk_id,
                    "vector": embedding,
                    "payload": payload,
                })
            except Exception as e:
                failures += 1
                logger.error("Error processing chunk %s: %s", chunk_id, e, exc_info=True)
                # Abort on too many consecutive failures
                if failures >= int(settings.embeddings_max_consecutive_failures_per_doc):
                    logger.error("Aborting document processing due to excessive embedding failures")
                    break
            # Time budget guard
            if (time.time() - started_at) > float(settings.embeddings_total_time_limit_secs):
                logger.error("Aborting document processing due to time limit exceeded")
                break

        return processed_chunks

    def index_document(self, document: Dict, force_delete: bool = True, max_chunks: Optional[int] = None) -> Dict[str, int]:
        """
        Index a document by processing it and storing embeddings
        
        Args:
            document: Dictionary containing text and metadata
            force_delete: If True, deletes existing Qdrant entries for the document URL before indexing
        """
        try:
            logger.debug("Indexing document: %s", document.get('url', 'unknown'))
            # Ensure collection exists before indexing
            try:
                self.qdrant.client.get_collection(settings.collection_name)
            except Exception:
                logger.debug("Collection not found, creating it now...")
                self.qdrant.create_collection()

            url = document.get('url')
            if url and force_delete:
                #logger.debug("Force delete enabled; deleting existing entries for: %s", url)
                self.delete_document(url)

            # Reset token counter and process document
            self._tokens_used = 0
            # Process document and generate embeddings
            processed_chunks = self.process_document(document, max_chunks=max_chunks)

            try:
                #logger.debug("Inserting %d vectors into Qdrant collection: %s", len(processed_chunks), self.qdrant.collection_name)
                success = self.qdrant.add_embeddings(processed_chunks)
                if not success:
                    raise Exception("Failed to add embeddings to Qdrant")
                #logger.debug("Successfully added %d embeddings to Qdrant", len(processed_chunks))
                return {"vectors_indexed": len(processed_chunks), "tokens_used": int(self._tokens_used)}
            except Exception as e:
                logger.exception("Error indexing embeddings to Qdrant: %s", e)
                raise
        except Exception as e:
            logger.exception("Error indexing document: %s", e)
            raise

    def remove_search_similar(self, query: str, limit: int = 5, query_filter: Optional[Any] = None) -> List[Dict]:
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
            try:
                from backend.core.config import settings as _s
            except Exception:
                _s = settings
            maxc = int(getattr(_s, "debug_log_truncate_chars", 500))
            if getattr(_s, "debug_verbose", False):
                q_snip = query if len(query) <= maxc else (query[:maxc] + "…")
                logger.debug("Searching for query: %s", q_snip)
            else:
                logger.debug("Searching for query (len=%d)", len(query or ""))
            query_embedding_result = self.generate_embeddings(query)
            # Handle new return format (embedding, magnitude, normalized, provider)
            if isinstance(query_embedding_result, tuple):
                query_embedding = query_embedding_result[0]
            else:
                # Legacy format (backward compatibility)
                query_embedding = query_embedding_result
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
            logger.debug("Found %d results for query", len(results))

            return [{
                "score": result.score,
                "payload": result.payload
            } for result in results]
        except Exception as e:
            logger.exception("Error searching Qdrant: %s", e)
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

    def index_chunks(self, chunks: List[Dict], force_delete: bool = True, max_chunks: Optional[int] = None) -> Dict[str, int]:
        """
        Index pre-chunked data into Qdrant. Generates embeddings for each chunk and wraps in Qdrant format.
        Args:
            chunks: List of dicts, each with at least 'text' and metadata (not pre-embedded)
            force_delete: If True, deletes existing Qdrant entries for the URL(s) before indexing
        """
        if not chunks:
            logger.debug("No chunks to index.")
            return

        #logger.debug("Indexing %d pre-chunked entries", len(chunks))

        try:
            self.qdrant.client.get_collection(settings.collection_name)
        except Exception:
            logger.debug("Collection not found, creating it now...")
            self.qdrant.create_collection()

        # Apply safety cap here as well; respect user-provided max if supplied
        effective_cap = int(settings.max_chunks_per_doc)
        if max_chunks is not None:
            try:
                user_cap = int(max_chunks)
                if user_cap > 0:
                    effective_cap = min(effective_cap, user_cap)
            except Exception:
                pass
        if len(chunks) > effective_cap:
            logger.warning("Capping pre-chunked inputs from %d to %d", len(chunks), effective_cap)
            chunks = chunks[:effective_cap]

        # Reset token counter; generate embeddings and wrap chunks
        self._tokens_used = 0
        # Resolve embedding spec once per batch to record the model name in payloads
        try:
            _emb_spec_chunks = resolve_embedding_spec(settings)
            _emb_model_name_chunks = _emb_spec_chunks.get("model")
            _emb_provider_chunks = _emb_spec_chunks.get("provider", "openai")
        except Exception:
            _emb_model_name_chunks = "text-embedding-3-small"
            _emb_provider_chunks = "openai"

        # Determine batch size based on embedding provider, with a safe default.
        try:
            if _emb_provider_chunks == "openai":
                batch_size = int(getattr(settings, "embedding_batch_size_openai", 25))
            elif _emb_provider_chunks == "gemini":
                batch_size = int(getattr(settings, "embedding_batch_size_gemini", 25))
            else:
                batch_size = int(getattr(settings, "embedding_batch_size_default", 25))
        except Exception:
            batch_size = 25
        if batch_size <= 0:
            batch_size = 1

        # DEBUG: Log the provider and chosen batch size for this pre-chunked indexing run.
        try:
            logger.debug(
                "[EMBEDDINGS] Using provider=%s batch_size=%s for pre-chunked inputs",
                _emb_provider_chunks,
                batch_size,
            )
        except Exception:
            pass

        points = []
        failures = 0
        started_at = time.time()

        # Process pre-chunked inputs in batches.
        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start: batch_start + batch_size]
            try:
                logger.debug(
                    "[EMBEDDINGS] Embedding pre-chunked batch starting_at=%s batch_len=%s",
                    batch_start,
                    len(batch),
                )
            except Exception:
                pass

            batch_texts: List[str] = []
            for chunk in batch:
                text = chunk.get("text", "")
                if not text:
                    batch_texts.append("")
                else:
                    batch_texts.append(text)

            try:
                if len(batch_texts) == 1:
                    embedding_result = self.generate_embeddings(batch_texts[0])
                    if isinstance(embedding_result, tuple):
                        # Single embedding with metadata
                        batch_embeddings = [embedding_result[0]]
                        batch_magnitudes = [embedding_result[1]]
                        batch_normalized = [embedding_result[2]]
                        batch_providers = [embedding_result[3]]
                    else:
                        # Legacy format (backward compatibility)
                        batch_embeddings = [embedding_result]
                        batch_magnitudes = [None]
                        batch_normalized = [False]
                        batch_providers = ['unknown']
                else:
                    embedding_result = self.generate_embeddings(batch_texts)  # type: ignore[arg-type]
                    if isinstance(embedding_result, tuple) and len(embedding_result) == 4:
                        # Batch embeddings with metadata
                        batch_embeddings, batch_magnitudes, batch_normalized, batch_providers = embedding_result
                    else:
                        # Legacy format (backward compatibility)
                        batch_embeddings = embedding_result
                        batch_magnitudes = [None] * len(batch_embeddings)
                        batch_normalized = [False] * len(batch_embeddings)
                        batch_providers = ['unknown'] * len(batch_embeddings)
            except Exception as e:
                failures += 1
                logger.error("Failed to embed batch starting at %d: %s", batch_start, e, exc_info=True)
                if failures >= int(settings.embeddings_max_consecutive_failures_per_doc):
                    logger.error("Aborting chunk indexing due to excessive embedding failures")
                    break
                if (time.time() - started_at) > float(settings.embeddings_total_time_limit_secs):
                    logger.error("Aborting chunk indexing due to time limit exceeded")
                    break
                continue

            for offset, chunk in enumerate(batch):
                i = batch_start + offset
                text = chunk.get("text", "")
                if not text:
                    logger.warning("Embeddings: Skipping empty chunk at index %d", i)
                    continue
                try:
                    # Ensure URL-derived helpers exist on the payload
                    u = chunk.get("url") or ""
                    if "url_lower" not in chunk:
                        chunk["url_lower"] = u.lower()
                    base = chunk.get("base_url")
                    if not base:
                        base = _strip_fragment(u)
                        chunk["base_url"] = base
                    if "base_url_lower" not in chunk:
                        chunk["base_url_lower"] = (base or "").lower()

                    # Best-effort: record the embedding model used, without overwriting
                    # any value that might already be present in the incoming chunk.
                    if _emb_model_name_chunks and "embedding_model" not in chunk:
                        chunk["embedding_model"] = _emb_model_name_chunks

                    try:
                        embedding = batch_embeddings[offset]
                        magnitude = batch_magnitudes[offset] if offset < len(batch_magnitudes) else None
                        normalized = batch_normalized[offset] if offset < len(batch_normalized) else False
                        provider = batch_providers[offset] if offset < len(batch_providers) else 'unknown'
                    except Exception:
                        embedding_result = self.generate_embeddings(text)
                        if isinstance(embedding_result, tuple):
                            embedding, magnitude, normalized, provider = embedding_result
                        else:
                            embedding, magnitude, normalized, provider = embedding_result, None, False, 'unknown'

                    # Add magnitude and normalization metadata if available
                    if magnitude is not None:
                        chunk["embedding_magnitude"] = magnitude
                    chunk["embedding_normalized"] = normalized
                    chunk["embedding_provider"] = provider

                    # Optional per-document token budget guard for pre-chunked inputs
                    try:
                        max_tokens_budget = int(getattr(settings, "embeddings_max_tokens_per_doc", 0) or 0)
                    except Exception:
                        max_tokens_budget = 0
                    if max_tokens_budget > 0 and self._tokens_used > max_tokens_budget:
                        logger.error(
                            "Aborting chunk indexing: embedding token budget exceeded (%s > %s)",
                            self._tokens_used,
                            max_tokens_budget,
                        )
                        break

                    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{u}-{i}"))
                    points.append({
                        "id": point_id,
                        "vector": embedding,
                        "payload": chunk
                    })
                except Exception as e:
                    failures += 1
                    logger.error("Failed to embed chunk %d: %s", i, e, exc_info=True)
                    if failures >= int(settings.embeddings_max_consecutive_failures_per_doc):
                        logger.error("Aborting chunk indexing due to excessive embedding failures")
                        break
                if (time.time() - started_at) > float(settings.embeddings_total_time_limit_secs):
                    logger.error("Aborting chunk indexing due to time limit exceeded")
                    break

        if force_delete:
            url_set = {point["payload"].get("url") for point in points}
            for url in url_set:
                if url:
                    self.delete_document(url)

        success = self.qdrant.add_embeddings(points)
        if not success:
            raise Exception("Failed to add embeddings to Qdrant")
        return {"vectors_indexed": len(points), "tokens_used": int(self._tokens_used)}
