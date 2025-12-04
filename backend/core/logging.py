"""Central logging configuration for the backend.

Keep this module lightweight and side-effect free. Import LOGGING_CONFIG
and apply it in the application entrypoint (e.g., main.py).
"""

import os
from pathlib import Path

# Ensure logs directory exists
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

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
            'level': 'INFO',
            'stream': 'ext://sys.stderr',
        },
        'file': {
            'class': 'concurrent_log_handler.ConcurrentRotatingFileHandler',
            'filename': str(log_dir / 'server.log'),
            'maxBytes': 5 * 1024 * 1024,  # 5MB per file
            'backupCount': 5,              # Keep 5 backup files
            'encoding': 'utf-8',
            'formatter': 'detailed',
            'level': 'DEBUG',
        },
        'error_file': {
            'class': 'concurrent_log_handler.ConcurrentRotatingFileHandler',
            'filename': str(log_dir / 'error.log'),
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
            'level': 'DEBUG',  # Capture all levels
            'propagate': True,
        },
        # Application logger
        'backend': {
            'handlers': ['console', 'file', 'error_file'],
            'level': 'DEBUG',
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
        # Silence verbose third-party DEBUG logs
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
