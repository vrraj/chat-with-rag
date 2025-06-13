from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    openai_api_key: str
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    embedding_model: str = "text-embedding-3-large"  # switched from text-embedding-3-small
    vector_size: int = 3072  # 1536 for text-embedding-3-small and 3072 for text-embedding-3-large; 
    chat_model: str = "gpt-4.o-mini"
    max_history_tokens: int = 4000
    collection_name: str = "docs_v3_large"  # switched from website_collection that is text-embedding-3-small
    html_chunk_size: int = 500
    html_chunk_overlap: int = 100
    pdf_chunk_size: Optional[int] = 500  # Will be determined by section length
    pdf_chunk_overlap: int = 100
    max_urls: int = 10
    default_chunk_size: int = 500
    default_chunk_overlap: int = 100
    

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }

settings = Settings()
