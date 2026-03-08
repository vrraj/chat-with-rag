"""Central logging configuration for the backend.

Import LOGGING_CONFIG and apply it in the application entrypoint (e.g., main.py).

Note: This module intentionally creates the ./logs directory at import time so
file handlers have a valid destination.
"""

import logging
import os
from pathlib import Path

# Ensure logs directory exists
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# --- Environment-driven log levels (minimal contract) ---
# LOG_LEVEL: overall logger level for the app/root (default: DEBUG)
# LOG_CONSOLE_LEVEL: console handler level (default: INFO)
# LOG_FILE_LEVEL: main file handler level (default: DEBUG)

def _env_level(name: str, default: str) -> str:
    """Return a valid logging level name from env, falling back to default."""
    raw = os.getenv(name, default)
    if raw is None:
        return default
    level = str(raw).strip().upper()
    # Accept standard Python logging level names.
    if level in logging._nameToLevel and level != "NOTSET":
        return level
    return default

ENV_LOG_LEVEL = _env_level("LOG_LEVEL", "DEBUG")
ENV_LOG_CONSOLE_LEVEL = _env_level("LOG_CONSOLE_LEVEL", "INFO")
ENV_LOG_FILE_LEVEL = _env_level("LOG_FILE_LEVEL", "DEBUG")

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            '()': 'uvicorn.logging.DefaultFormatter',
            'fmt': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s [in %(pathname)s:%(lineno)d]',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
            'level': ENV_LOG_CONSOLE_LEVEL,
            'stream': 'ext://sys.stderr',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(log_dir / 'server.log'),
            'maxBytes': 5 * 1024 * 1024,  # 5MB per file
            'backupCount': 5,              # Keep 5 backup files
            'encoding': 'utf-8',
            'formatter': 'detailed',
            'level': ENV_LOG_FILE_LEVEL,
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(log_dir / 'warnings_and_errors.log'),
            'maxBytes': 5 * 1024 * 1024,
            'backupCount': 5,
            'encoding': 'utf-8',
            'formatter': 'detailed',
            'level': 'WARNING',
        },
    },
    'loggers': {
        # Root logger - catches everything
        '': {
            'handlers': ['console', 'file', 'error_file'],
            'level': ENV_LOG_LEVEL, 
            'propagate': True,
        },
        # Application logger
        'backend': {
            'handlers': ['console', 'file', 'error_file'],
            'level': ENV_LOG_LEVEL,
            'propagate': False,
        },
        # Third-party loggers
        'trafilatura': {
            'level': 'WARNING',
            'handlers': ['file', 'error_file'],
            'propagate': False
        },
        # Uvicorn loggers
        'uvicorn': {
            'handlers': ['console', 'file', 'error_file'],
            'level': 'INFO',
            'propagate': False,
        },
        'uvicorn.error':  {
            'handlers': ['console', 'file', 'error_file'],
            'level': 'INFO',
            'propagate': False,
        },
        'uvicorn.access': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'openai': {
            'handlers': ['file', 'error_file'],
            'level': 'WARNING',
            'propagate': False,
        },
        'httpx': {
            'handlers': ['file', 'error_file'],
            'level': 'WARNING',
            'propagate': False,
        },
        'httpcore': {
            'handlers': ['file', 'error_file'],
            'level': 'WARNING',
            'propagate': False,
        },
    }
}
