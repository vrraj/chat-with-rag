
# run.py
# Application startup script - runs uvicorn

import uvicorn
import os
from logging import getLogger
from logging.config import dictConfig
from logging import INFO

# Import the configuration from logging.py file
from backend.core.logging import LOGGING_CONFIG

# The logs directory must exist before a file handler can write to it
if not os.path.exists("logs"):
    os.makedirs("logs")

# Apply the logging configuration from the dictionary
dictConfig(LOGGING_CONFIG)

# Optional: This line is not strictly necessary but demonstrates getting a logger
# and checking its level.
logger = getLogger("backend")
logger.info("Application starting up...")

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["backend"],
	reload_includes=[".env"],
        log_config=LOGGING_CONFIG, # Pass the imported configuration here
        log_level=INFO # Set the overall logging level.
    )

