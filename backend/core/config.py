"""Configuration settings for the application."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings


# -----------------------------------------------------------------------------
# Frontend & Indexing Configuration Models
# -----------------------------------------------------------------------------
class MediaWikiConfig(BaseModel):
    """Configuration for MediaWiki API interactions."""

    max_chunks: int = 0
    skip_sections: List[str] = [
        "References",
        "External links",
        "See also",
        "Further reading",
    ]
    estimate: bool = True
    force_delete: bool = False
    api_url: str = "https://en.wikipedia.org/w/api.php"
    user_agent: str = "WebsiteChatAgent/0.1 (contact@example.com)"

    @field_validator("max_chunks")
    def validate_max_chunks(cls, v):
        if v < 0:
            raise ValueError("max_chunks cannot be negative")
        return v

    @field_validator("api_url")
    def validate_api_url(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("api_url must start with http:// or https://")
        return v


class HTMLConfig(BaseModel):
    """Configuration for HTML content indexing."""

    max_chunks: int = 0  # 0 means no limit
    skip_sections: List[str] = [
        "References",
        "External links",
        "See also",
        "Further reading",
    ]
    estimate: bool = True
    force_delete: bool = False

    @field_validator("max_chunks")
    def validate_max_chunks(cls, v):
        if v < 0:
            raise ValueError("max_chunks cannot be negative")
        return v


class PDFConfig(BaseModel):
    """Configuration for PDF content indexing."""

    max_chunks: int = 0
    # Default sections to skip when indexing PDFs, kept in sync with frontend defaults
    # and PDFInput schema where possible.
    skip_sections: List[str] = [
        "References",
        "External links",
        "Further reading",
        "Notes",
        "See Also",
        "Acknowledgements",
    ]
    estimate: bool = True
    force_delete: bool = False

    @field_validator("max_chunks")
    def validate_max_chunks(cls, v):
        if v < 0:
            raise ValueError("max_chunks cannot be negative")
        return v



class FrontendConfig(BaseModel):
    html: HTMLConfig = HTMLConfig()
    pdf: PDFConfig = PDFConfig()
    mediawiki: MediaWikiConfig = MediaWikiConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary with camelCase keys for the frontend."""

        def to_camel_case(snake_str: str) -> str:
            components = snake_str.split("_")
            return components[0] + "".join(x.title() for x in components[1:])

        def convert_dict(d):
            if isinstance(d, BaseModel):
                d = d.dict()
            if isinstance(d, dict):
                return {to_camel_case(k): convert_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_dict(i) for i in d]
            return d

        return convert_dict(self.dict())


# -----------------------------------------------------------------------------
# Main application Settings
# NOTE: Settings are intentionally kept flat (no nested sub-models)
# -----------------------------------------------------------------------------

class Settings(BaseSettings):
    # -------------------------------------------------------------------------
    # 1) Core API & frontend exposure
    # -------------------------------------------------------------------------
    openai_api_key: str
    openai_api_base: str = "https://api.openai.com/v1"  # for future use
    gemini_api_key: str
    gemini_api_base: str = "https://generativelanguage.googleapis.com/v1beta/openai/"  # for future use

    frontend: FrontendConfig = FrontendConfig()  # Configuration for frontend forms and document indexing (HTML/PDF/MediaWiki)

    # -------------------------------------------------------------------------
    # 2) Vector search & retrieval (Qdrant)
    # -------------------------------------------------------------------------
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    #collection_name: str = "document_index"  # collection name
    collection_name: str = "document_index_gemini"  # collection name

    # Vector/search shape & retrieval knobs
    # Qdrant vector size (must match embedding dimension)
    vector_size: int = 1536
    top_k: int = 8  # Recall: Number of documents to retrieve
    score_threshold: float = 0.35  # Precision: Minimum vector similarity score
    exact_match: bool = False  # use HNSW for faster search as opposed to ANN. Adjust results are not optimal


    # -------------------------------------------------------------------------
    # 3) Embeddings
    # -------------------------------------------------------------------------
    # Embedding provider: "openai" or "gemini" (used for docs + queries)
    embedding_model: str = "gemini"

    # Default OpenAI embedding model
    openai_embedding_model: str = "text-embedding-3-small"

    # Embedding profile key (must match model_registry)
    # e.g. "openai:embed_small", "gemini:embed"
    # Model and dimensions should stay in sync with vector_size.
    # embedding_model_key: str = "openai:embed_small"


    # Gemini embeddings
    gemini_embedding_model: str = "gemini-embedding-001"
    # Must match vector_size and registry capabilities["dimensions"]
    gemini_embedding_dimensions: int = 1536
    # Whether to L2-normalize Gemini embeddings client-side (adapter/native paths).
    # This should be applied consistently for both indexing and query embeddings.
    gemini_embedding_normalize: bool = True
    
    # Gemini embedding task types for different use cases
    gemini_embed_type_documents: str = "RETRIEVAL_DOCUMENT"  # For indexing documents
    gemini_embed_type_query: str = "RETRIEVAL_QUERY"  # For user search queries

    # Cost basis for per‑MM pricing
    cost_basis_tokens: int = 1_000_000

    # Embedding batch sizes (number of chunks sent in a single embeddings.create call)
    embedding_batch_size_default: int = 30
    embedding_batch_size_openai: int = 30
    embedding_batch_size_gemini: int = 30

    # -------------------------------------------------------------------------
    # 3B) LLM model profiles (registry keys)
    # -------------------------------------------------------------------------
    # Model keys are stable aliases that map to provider/model/pricing in model_registry.py.
    # Examples: "openai:fast", "openai:best", "gemini:fast", "openai:embed_small".

    # Embeddings profile key (should align with `embedding_model` provider selector).
    #embedding_model_key: str = "openai:embed_small"
    embedding_model_key: str = "gemini:embed"
    # Stage model profile keys
    rewrite_model_key: str = "openai:fast"
    rerank_model_key: str = "openai:fast"
    summarizer_model_key: str = "openai:fast"

    inference_model_key: str = "openai:fast"

    # If unset, tools synthesis inherits the inference model_key.
    tools_synth_model_key: str | None = None

    # -------------------------------------------------------------------------
    # 4) Re-ranker
    # -------------------------------------------------------------------------
    re_ranker_model: str = "gpt-4o-mini"  # use for faster, lower-cost inference

    re_ranker_max_output_tokens: int = 50
    re_ranker_input_rows: int = 5
    re_ranker_temperature: float = 0.3

    # Policy notes:
    # - If retrieval returns < re_ranker_input_rows, chat_manager should skip rerank.
    # - If top1 is strong and the margin is large, chat_manager may skip rerank.
    # - Exact-match fast path is optional.
    rerank_clear_winner_min_top1: float = 0.60
    rerank_clear_winner_min_delta: float = 0.15
    rerank_exact_match_min_score: float = 0.80

    # -------------------------------------------------------------------------
    # 5) Summarizer
    # -------------------------------------------------------------------------
    summarizer_model: str = "gpt-4o-mini"  # use for faster, lower-cost inference
    summarizer_max_input_tokens: int = 400  # set an int to limit input tokens
    summarizer_max_output_tokens: int = 200  # set an int to limit output tokens
    summarizer_temperature: float = 0.3

    # -------------------------------------------------------------------------
    # 6) Inference model
    # -------------------------------------------------------------------------
    inference_model: str = "gpt-4o-mini"  # use for faster, lower-cost inference
    inference_tools_synthesis_model: str = "gpt-4o-mini" # Deprecated: Inference with Tool Synthsis will always use inference model to maintain consistency at inference stages 

    # -------------------------------------------------------------------------
    # 6B) Prompt Registry (YAML)
    # -------------------------------------------------------------------------
    # Required: path to the prompt registry YAML for inference stage-1 prompt construction.
    inference_prompt_registry_path: str = "prompts/prompt_registry.yaml"
    # Optional per-request override: params["prompt_domain"]. When unset, fall back to this default.
    prompt_domain_default: str = ""

    inference_temperature: float = 0.4  # Decoding temperature
    inference_top_p: float = 0.7  # Nucleus sampling top-p

    # --- Inference context control --- Number of reranked rows (retrieved) to include in inference prompt as input context
    inference_context_rows: int = 4

    max_inference_output_tokens: int = 500
    tools_synth_max_output_tokens: int = 600

    # to include reasoning for the inference_model, set the inference_reasoning_effort and inference_reasoning_model
    # inference_reasoning_effort: "low" | "medium" | "high"
    # inference_reasoning_model: True to use reasoning

    inference_reasoning_effort: str = "low"
    inference_reasoning_model: bool = False

    debug_thoughts: bool = True # Gemini specific flag to display reasoning - the llm_handler will ignore this for other models

    enable_tools: bool = True  # Enable agent-style tool calls (UI can override per-turn)
    max_tool_passes: int = 2  # Maximum number of tool loops to be called from LLM generated output for a single turn. This it to prevent runaway tool calls

    # Tools that should receive document snippets (reranked context) as `existing_context`.
    # Most tools (e.g., get_weather, closest_airports) should NOT be listed here.
    tools_with_document_context: list[str] = [
        # "quote_from_docs",
        # "find_in_sources",
        # "cite_sources",
    ]

    # -------------------------------------------------------------------------
    # 7) Conversation context trimming
    # -------------------------------------------------------------------------
    max_history_tokens: int = 4000  # Max tokens of prior chat retained when building context

    # Conversation context strategy:
    # - chat_history_window_turns: summarized older turns
    # - raw_tail_turns: most-recent turns kept verbatim
    chat_history_window_turns: int = 2
    raw_tail_turns: int = 2

    enable_query_rewrite: bool = True

    # Default toggle for automatic web search to populate WEB SEARCH RESULTS / web_context.
    # Request-level flags can override this per turn.
    use_web_search: bool = False

    # How many most‑recent turns the rewriter sees (can differ from raw_tail_turns if desired)
    rewrite_tail_turns: int = 2

    # Cache rewrites for a short time to avoid repeat calls on identical context
    rewrite_cache_ttl_s: int = 300

    # Summary cache idle TTL (seconds)    # Optional: idle eviction TTL for summary cache (seconds). Defaults to 3600 if unset.
    summary_cache_idle_ttl_seconds: int | None = 3600

    # --- UI display toggles ---
    # Whether to append the Sources: block + structured sources for the main chat UI.
    display_sources_for_chat: bool = True
    # Whether to append the Sources: block + structured sources for embed-chat.
    # Default False so embeds can opt out of inline sources while sharing the same backend.
    display_sources_for_embed: bool = False

    # -------------------------------------------------------------------------
    # 9) Initial origin/host-based protection for critical FastAPI routes
    # -------------------------------------------------------------------------
    # Comma-separated list of allowed Origin header values.
    # Example for dev + prod:
    #   "http://localhost:8000,https://chat-with-rag.com"
    # When empty/None, origin-based checks are disabled.
    allowed_origins: Optional[str] = "http://localhost:8000,http://chat-with-rag:8000"

    # Comma-separated list of allowed hosts (hostname or hostname:port) for
    # requests hitting critical API routes such as /chat and ingestion.
    # Example:
    #   "localhost:8000,chat-with-rag.com"
    # When empty/None, host-based checks are disabled.
    allowed_hosts: Optional[str] = "localhost:8000,chat-with-rag:8000"

    # -------------------------------------------------------------------------
    # 10) Chunking & ingestion (shared defaults + per-source toggles)
    # -------------------------------------------------------------------------
    # NOTE: When setting chunk sizes, consider provider limits:
    # - OpenAI: Max 8,191 tokens per text, variable tokens per request (check your tier)
    # - Gemini: Max 2,048 tokens per text (8,000 on newer models), 20,000 tokens per request
    # Refer to provider documentation for current limits and pricing tiers.
    max_urls: int = 10 # legacy - remove when refactoring
    default_chunk_size: int = 500
    default_chunk_overlap: int = 100
    reranker_chunk_size: int = 500

    # HTML extraction configuration
    html_chunk_size: int = 500
    html_chunk_overlap: int = 100
    html_index_tables: bool = True  # Optional: if True, index tables as structured payloads
    html_table_rows_per_chunk: int = 12  # Optional: number of table rows per chunk
    html_drop_tables_from_prose: bool = True  # Optional: if True, remove ALL <table> elements from the prose extraction path
    html_skip_sections: Optional[List[str]] = None  # Optional: list of section names to skip (e.g., "References", "See also")

    # PDF extraction configuration
    pdf_chunk_size: Optional[int] = 500  # chunk size for prose chunking
    pdf_chunk_overlap: int = 100
    pdf_skip_sections: Optional[List[str]] = None  # e.g., ["References", "See also"]

    pdf_use_pymupdf4llm: bool = True  # Use pymupdf4llm extractor (vs legacy pymupdf)

    pdf_header_footer_filter: bool = True  # Filter repeated headers/footers
    pdf_multicolumn_sort: bool = False  # Attempt multi-column reading order

    # --- PDF table indexing (structured path) ---
    pdf_index_tables: bool = True  # Emit tables as structured markdown payloads (separate from prose)
    pdf_table_rows_per_chunk: int = 12  # Number of data rows per emitted table chunk
    pdf_repeat_table_header: bool = True  # Repeat header row in each table chunk
    pdf_table_min_rows: int = 1  # Minimum data rows required to index a detected table
    pdf_drop_tables_from_prose: bool = True  # Remove tables from prose extraction path (keep structured payloads)

    # Back-compat alias used by older code paths (prefer pdf_header_footer_filter going forward)
    header_footer_filter: bool = True

    mediawiki_chunk_size: int = 500
    mediawiki_chunk_overlap: int = 100

    wiki_mode: str = "parsoid"
    wiki_index_tables: bool = True  # Optional: if True, index tables as structured payloads
    wiki_table_rows_per_chunk: int = 12  # Optional: number of table rows per chunk
    wiki_drop_tables_from_prose: bool = True  # Optional: if True, remove ALL <table> elements from the prose extraction path

    # Wikipedia / MediaWiki API configuration
    wiki_api_url: str = "https://en.wikipedia.org/w/api.php"
    wiki_user_agent: str = "WebsiteChatAgent/0.1 (contact: set-in-settings)"
    wiki_timeout_secs: int = 15
    wiki_max_retries: int = 5

    # -------------------------------------------------------------------------
    # 10) Embedding & indexing safeguards
    # -------------------------------------------------------------------------
    max_chunks_per_doc: int = 500  # Embedding safety: hard cap to prevent runaway loops
    embeddings_max_retries: int = 5  # Retry attempts per embedding call
    embeddings_initial_backoff_secs: float = 1.0  # Initial backoff for retries
    embeddings_max_consecutive_failures_per_doc: int = 20  # Abort document after too many failures
    embeddings_total_time_limit_secs: int = 300  # Abort document if processing exceeds this time
    embeddings_call_delay_secs: float = 0.0  # Optional pacing between embedding calls
    embeddings_max_tokens_per_doc: int = 200000  # Optional per-document token budget guard

    # Indexing safeguards:  When True, ingestion routes will first check whether a document
    # is already present in Qdrant (by matching `url_lower`). User need to confirm "Force Delete" to override
    check_document_indexed: bool = True

    # -------------------------------------------------------------------------
    # 11) Debug / logging controls
    # -------------------------------------------------------------------------
    debug_verbose: bool = False  # gates noisy logs (prompts, raw outputs)
    debug_log_keys: bool = False  # gates any API key suffix logging
    debug_log_truncate_chars: int = 200  # max chars to print when debug_verbose is True
    show_processing_steps: bool = True  # controls whether intermediate SSE processing stages are emitted

    # -------------------------------------------------------------------------
    # 12) Pydantic settings model config (preserved)
    # -------------------------------------------------------------------------
    # Shared directory for PDF files that can be referenced by filename only
    # shared_pdf_directory: str = Field(env="SHARED_PDF_DIRECTORY", default="/tmp/shared_pdfs")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        # Allow extra env vars (e.g., feature flags) without validation errors.
        "extra": "ignore",
    }


# Initialize settings after all classes are defined
settings = Settings()
