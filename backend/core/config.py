from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    openai_api_key: str
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4.1-mini-2025-04-14"
    max_history_tokens: int = 4000
    collection_name: str = "website_collection"
    html_chunk_size: int = 500
    html_chunk_overlap: int = 100
    pdf_chunk_size: Optional[int] = 500  # Will be determined by section length
    pdf_chunk_overlap: int = 100
    max_urls: int = 10
    default_chunk_size: int = 500
    default_chunk_overlap: int = 100
    vector_size: int = 1536  # default for OpenAI; changeable for other models

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }

settings = Settings()
