from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    openai_api_key: str
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # embedding_model: str = "text-embedding-3-large"  # use for higher-quality embeddings
    # vector_size: int = 3072  # use with text-embedding-3-large
    # collection_name: str = "docs_v3_large"  # collection name for large embedding model

    embedding_model: str = "text-embedding-3-small"  # use for faster, lower-cost embeddings
    vector_size: int = 1536  # use with text-embedding-3-small
    collection_name: str = "website_collection"  # collection name for small embedding model

    chat_model: str = "gpt-4.o-mini"
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


    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }

settings = Settings()
