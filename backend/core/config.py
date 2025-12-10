"""
Configuration settings for the application.

This module defines all configuration settings, including:
- Frontend form defaults
- Model configurations
- API endpoints
- System behaviors

Frontend Configuration:
---------------------
The frontend configuration is exposed via /api/config/api-defaults and controls:
- Default values for form fields
- Behavior of different indexing options (HTML, PDF, MediaWiki)
- API endpoints and user agents for external services

To modify these values:
1. Update the default values in the respective config classes below
2. The changes will be automatically available through the API
3. The frontend will pick up the new defaults on page refresh
"""

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Any

class HTMLConfig(BaseModel):
    """Configuration for HTML content indexing."""
    max_chunks: int = 0  # 0 means no limit
    skip_sections: List[str] = ["References", "External links", "See also", "Further reading"]
    estimate: bool = True
    force_delete: bool = False
    
    @field_validator('max_chunks')
    def validate_max_chunks(cls, v):
        if v < 0:
            raise ValueError("max_chunks cannot be negative")
        return v

class PDFConfig(BaseModel):
    """Configuration for PDF content indexing."""
    max_chunks: int = 0
    estimate: bool = True
    force_delete: bool = False
    
    @field_validator('max_chunks')
    def validate_max_chunks(cls, v):
        if v < 0:
            raise ValueError("max_chunks cannot be negative")
        return v

class MediaWikiConfig(BaseModel):
    """Configuration for MediaWiki API interactions."""
    max_chunks: int = 0
    skip_sections: List[str] = ["References", "External links", "See also", "Further reading"]
    estimate: bool = True
    force_delete: bool = False
    api_url: str = "https://en.wikipedia.org/w/api.php"
    user_agent: str = "WebsiteChatAgent/0.1 (contact@example.com)"
    
    @field_validator('max_chunks')
    def validate_max_chunks(cls, v):
        if v < 0:
            raise ValueError("max_chunks cannot be negative")
        return v
        
    @field_validator('api_url')
    def validate_api_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("api_url must start with http:// or https://")
        return v

class FrontendConfig(BaseModel):
    html: HTMLConfig = HTMLConfig()
    pdf: PDFConfig = PDFConfig()
    mediawiki: MediaWikiConfig = MediaWikiConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary with camelCase keys for the frontend."""
        def to_camel_case(snake_str: str) -> str:
            components = snake_str.split('_')
            return components[0] + ''.join(x.title() for x in components[1:])
        
        def convert_dict(d):
            if isinstance(d, BaseModel):
                d = d.dict()
            if isinstance(d, dict):
                return {to_camel_case(k): convert_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_dict(i) for i in d]
            return d
            
        return convert_dict(self.dict())

class Settings(BaseSettings):
    openai_api_key: str
    openai_api_base: str = "https://api.openai.com/v1"  # for future use
    frontend: FrontendConfig = FrontendConfig()  # Initialize with default values
    # Vector Search Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333 # 
    vector_size: int = 1536  # use with text-embedding-3-small
    exact_match: bool = False # use HNSW for faster search as opposed to ANN. Adjust results are not optimal
    collection_name: str = "document_index"  # collection name
    top_k: int = 8 # Recall: Number of documents to retrieve
    score_threshold: float = .35 # Precision: Minimum vector similarity score

    # embedding_model: str = "text-embedding-3-large"  # use for higher-quality embeddings
    # vector_size: int = 3072  # use with text-embedding-3-large
    # collection_name: str = "docs_v3_large"  # collection name for large embedding model

    # Embedding configuration
    embedding_model: str = "text-embedding-3-small"  # use for faster, lower-cost embeddings
    # Cost configuration (USD per 1,000,000 tokens)
    embedding_cost_per_MM_tokens: float = 0.02
    # Cost basis (all rates are per this many tokens)
    cost_basis_tokens: int = 1_000_000
    # Re-ranker configuration
    re_ranker_model: str = "gpt-4o-mini"  # use for faster, lower-cost inference
    # Cost configuration (USD per 1,000,000 tokens)
    re_ranker_cost_per_MM_tokens_input: float = 0.15
    re_ranker_cost_per_MM_tokens_output: float = 0.60
    re_ranker_cost_per_MM_tokens_cached_input: float = 0.075
    re_ranker_max_output_tokens: int = 128
    re_ranker_input_rows: int = 5
    re_ranker_temperature: float = 0.3

    # --- Rerank decision policy (used by chat_manager) ---
    # Skip rerank if retrieved results are fewer than this many. We reuse
    # re_ranker_input_rows as the minimum candidate pool for rerank.
    # If retrieval returns < re_ranker_input_rows, chat_manager should skip rerank.

    # Criteria to skip rerank when there is a clear winner by vector score.
    # If top1 score is at least this value AND the margin (top1 - top5) is
    # at least rerank_clear_winner_min_delta, skip rerank.
    rerank_clear_winner_min_top1: float = 0.65
    rerank_clear_winner_min_delta: float = 0.15

    # Optional exact-match fast path (string/ID/hash style queries) — if your
    # retrieval marks an item as an exact match and its score exceeds this,
    # chat_manager can skip rerank. Wiring this is optional; safe default.
    rerank_exact_match_min_score: float = 0.80

    # Summarizer model configuration
    summarizer_model: str = "gpt-4o-mini"  # use for faster, lower-cost inference
    # Cost configuration (USD per 1,000,000 tokens)
    summarizer_max_input_tokens :int = 400  # set an int to limit input tokens
    summarizer_max_output_tokens: int = 200 # set an int to limit output tokens
    summarizer_cost_per_MM_tokens_input: float = 0.15
    summarizer_cost_per_MM_tokens_output: float = 0.60
    summarizer_cost_per_MM_tokens_cached_input: float = 0.075
    summarizer_temperature: float = 0.3
    # Inference Model Configurations
    inference_model: str = "gpt-4o-mini"  # use for faster, lower-cost inference
    # Inference decoding parameters
    inference_temperature: float = 0.4
    inference_top_p: float = 0.9

    # Debug / logging controls
    debug_verbose: bool = True            # gates noisy logs (prompts, raw outputs)
    debug_log_keys: bool = False         # gates any API key suffix logging
    debug_log_truncate_chars: int = 4000  # max chars to print when debug_verbose is True


    # Cost configuration (USD per 1,000,000 tokens)
    inference_cost_per_MM_tokens_input: float = 0.15
    inference_cost_per_MM_tokens_output: float = 0.60
    inference_cost_per_MM_tokens_cached_input: float = 0.075
    max_inference_output_tokens: int = 300
    inference_reasoning_effort: str = "low"
    inference_reasoning_model: bool = False
    # Tool use (agent-style) default: off; UI can override per-turn
    enable_tools: bool = True
    max_tool_passes: int = 2 # Maximum number of tool loops to be called from LLM generated output for a single turn. This it to prevent runaway tool calls

    # Maximum number of tokens from prior chat messages to retain when building
    # conversation context (used when trimming history in backend/main.py).
    max_history_tokens: int = 4000
    # Conversation context strategy
    # - chat_history_window_turns: How many **older** turns (user+assistant pairs) to summarize
    #   before the most-recent verbatim tail is added to the prompt. One "turn" = 2 messages.
    # - raw_tail_turns: How many **most-recent** turns to include verbatim in the prompt
    #   (preserves nuance like pronouns and references). This setting only changes behavior once
    #   the handler is wired to use it; safe to keep as a no-op until then.
    chat_history_window_turns: int = 5
    raw_tail_turns: int = 2

    # --- Query rewrite (for retrieval) ---
    # Master switch: when False, retrieval uses the original user message (current behavior)
    enable_query_rewrite: bool = True
    # Small, cost‑effective model for rewrite (keeps latency/$$ low)
    rewrite_model: str = "gpt-4o-mini"
    rewrite_temperature: float = 0.3
    # Cost configuration (USD per 1,000,000 tokens) — mirrors summarizer by default
    rewrite_cost_per_MM_tokens_input: float = 0.15
    rewrite_cost_per_MM_tokens_output: float = 0.60
    rewrite_cost_per_MM_tokens_cached_input: float = 0.075
    # Require sufficient confidence to accept a rewrite; otherwise fall back to original
    rewrite_confidence_threshold: float = 0.65
    # Keep rewrite outputs tiny and structured (JSON)
    rewrite_max_output_tokens: int = 80
    # How many most‑recent turns the rewriter sees (can differ from raw_tail_turns if desired)
    rewrite_tail_turns: int = 2
    # Cache rewrites for a short time to avoid repeat calls on identical context
    rewrite_cache_ttl_s: int = 300
    # Optional future: per‑domain policy name (unused until wired)
    # rewrite_domain_policy: str = "default"
    # Summary cache idle TTL (seconds) - will evict per-namespace summaries that haven't been used in this long
    summary_cache_idle_ttl_seconds: int = 1800

    # Chunking configuration
    html_chunk_size: int = 500
    html_chunk_overlap: int = 100
    pdf_chunk_size: Optional[int] = 500  # Will be determined by section length
    pdf_chunk_overlap: int = 100
    max_urls: int = 10
    default_chunk_size: int = 500
    default_chunk_overlap: int = 100
    reranker_chunk_size: int = 500
    mediawiki_chunk_size: int = 500
    mediawiki_chunk_overlap: int = 100
    
    # switch between legacy pymupdf extractor vs new pymupdf4llm extractor
    pdf_use_pymupdf4llm: bool = True

    # Wikipedia / MediaWiki API configuration
    wiki_api_url: str = "https://en.wikipedia.org/w/api.php"
    wiki_user_agent: str = "WebsiteChatAgent/0.1 (contact: set-in-settings)"
    wiki_timeout_secs: int = 15
    wiki_max_retries: int = 5

    # Embedding safety limits
    max_chunks_per_doc: int = 500  # Hard cap to prevent runaway embedding loops
    embeddings_max_retries: int = 5  # Retry attempts per embedding call
    embeddings_initial_backoff_secs: float = 1.0  # Initial backoff for retries
    embeddings_max_consecutive_failures_per_doc: int = 20  # Abort document after too many failures
    embeddings_total_time_limit_secs: int = 300  # Abort document if processing exceeds this time
    embeddings_call_delay_secs: float = 0.0  # Optional pacing between embedding calls
    embeddings_max_tokens_per_doc: int = 200000  # Optional per-document token budget guard

    # Indexing safeguards
    # When True, ingestion routes will first check whether a document
    # is already present in Qdrant (by matching `url_lower`). If found,
    # they will return a confirmation-required response instead of
    # re-indexing immediately.
    check_document_indexed: bool = True


    # Shared directory for PDF files that can be referenced by filename only
    # shared_pdf_directory: str = Field(env="SHARED_PDF_DIRECTORY", default="/tmp/shared_pdfs")
    
    model_config = {
            "env_file": ".env",
            "env_file_encoding": "utf-8",
            "case_sensitive": False,
            # Allow extra environment variables without raising ValidationError.
            # This lets you keep feature flags like PIPELINE_* in the .env file
            # even if they are not defined as Settings fields.
            "extra": "ignore",
        }

# Initialize settings after all classes are defined
settings = Settings()
