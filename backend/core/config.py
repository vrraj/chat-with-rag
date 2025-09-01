from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    openai_api_key: str
    openai_api_base: str = "https://api.openai.com/v1" # for future use
    # Vector Search Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333 # 
    vector_size: int = 1536  # use with text-embedding-3-small
    collection_name: str = "website_collection"  # collection name for small embedding model
    top_k: int = 5
    score_threshold: float = 0.35

    # embedding_model: str = "text-embedding-3-large"  # use for higher-quality embeddings
    # vector_size: int = 3072  # use with text-embedding-3-large
    # collection_name: str = "docs_v3_large"  # collection name for large embedding model

    # Embedding configuration
    embedding_model: str = "text-embedding-3-small"  # use for faster, lower-cost embeddings
    # Cost configuration (USD per 1,000,000 tokens)
    embedding_cost_per_MM_tokens: float = 0.01
    # Re-ranker configuration
    re_ranker_model: str = "gpt-4o-mini"  # use for faster, lower-cost inference
    # Cost configuration (USD per 1,000,000 tokens)
    re_ranker_cost_per_MM_tokens_input: float = 0.15
    re_ranker_cost_per_MM_tokens_output: float = 0.60
    # Inference Model COnfigurations
    inference_model: str = "gpt-4o-mini"  # use for faster, lower-cost inference
    # Cost configuration (USD per 1,000,000 tokens)
    inference_cost_per_MM_tokens_input: float = 0.15
    inference_cost_per_MM_tokens_output: float = 0.60
    max_output_tokens: int = 300
    

    max_history_tokens: int = 4000
    html_chunk_size: int = 500
    html_chunk_overlap: int = 100
    pdf_chunk_size: Optional[int] = 500  # Will be determined by section length
    pdf_chunk_overlap: int = 100
    max_urls: int = 10
    default_chunk_size: int = 500
    default_chunk_overlap: int = 100

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

    # Indexing safeguards
    # When True, ingestion routes will first check whether a document
    # is already present in Qdrant (by matching `url_lower`). If found,
    # they will return a confirmation-required response instead of
    # re-indexing immediately.
    check_document_indexed: bool = True


    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }

settings = Settings()
