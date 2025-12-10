#!/usr/bin/env python3
"""
Batch Document Processor for RAG Pipeline

This script processes multiple documents in batch mode by sending them to the API endpoint.
It supports various document types (HTML, PDF, MediaWiki) and provides detailed logging.

Features:
- Process multiple documents in a single batch
- Support for different document types (HTML, PDF, MediaWiki)
- Configurable chunking and processing options
- Detailed logging and error handling
- Retry mechanism for failed requests
- Timestamped output files

Input Format (JSON):
{
    "items": [
        {
            "url": "https://example.com",
            "doc_type": "html|pdf|mediawiki",
            "skip_sections": ["References", "External links"],
            "user_agent": "Custom User Agent (optional)",
            "api_url": "Custom API URL (for MediaWiki, optional)"
        }
    ],
    "max_chunks": 100,          // Optional: Limit chunks per document
    "estimate": true,           // Optional: Run in estimation mode
    "force_delete": false       // Optional: Force re-indexing
}

Usage:
    # Show help
    python -m scripts.batch.process_docs --help
    
    # Create a sample input file
    python -m scripts.batch.process_docs --create-sample
    
    # Process documents (estimation mode by default)
    python -m scripts.batch.process_docs
    
    # Process with custom input file
    python -m scripts.batch.process_docs --input path/to/input.json
    
    # Process for real (not estimation)
    python -m scripts.batch.process_docs --no-estimate
    
    # Force re-indexing of existing documents
    python -m scripts.batch.process_docs --force-delete
    
    # Limit maximum chunks per document
    python -m scripts.batch.process_docs --max-chunks 50
    
    # Use custom API endpoint
    python -m scripts.batch.process_docs --api-url http://localhost:8000/batch/process_docs

Output:
    - Results are saved in JSONL format (one JSON object per line)
    - Filename format: batch_result_YYYYMMDD_HHMMSS.jsonl
    - Output directory: ./scripts/batch/output/

Example Output Line:
    {
        "timestamp": "2025-09-22T23:45:00.123456",
        "input_file": "/path/to/input.json",
        "estimate": true,
        "force_delete": false,
        "max_chunks": 100,
        "results": [
            {
                "url": "https://example.com",
                "doc_type": "html",
                "status": "success",
                "result": {
                    "message": "Estimate only",
                    "chunks_planned": 42,
                    "tokens_used": 12345,
                    "embedding_cost": 0.00012345
                }
            }
        ]
    }
"""

import json
import logging
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import click
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add project root to path to import core modules
sys.path.append(str(Path(__file__).parent.parent))

# Import logging configuration
from backend.core.logging import LOGGING_CONFIG

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("backend.batch")

# Add a custom log level for progress reporting
PROGRESS_LEVEL = 25
logging.addLevelName(PROGRESS_LEVEL, "PROGRESS")

def progress(self, message, *args, **kws):
    if self.isEnabledFor(PROGRESS_LEVEL):
        self._log(PROGRESS_LEVEL, message, args, **kws)

logging.Logger.progress = progress

# Default API endpoint
DEFAULT_API_URL = "http://localhost:8000/batch/process_docs"

# Default input/output directories
SCRIPT_DIR = Path(__file__).parent.absolute()
DEFAULT_INPUT_DIR = SCRIPT_DIR / "input"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"

# Create directories if they don't exist
DEFAULT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def create_session() -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def log_response(url: str, response: Dict[str, Any]) -> None:
    """Log the API response with appropriate log level and format matching the API response model."""
    status = response.get("status", "unknown").lower()
    
    # Handle 'already indexed' case
    if response.get("already_indexed", False):
        message = (
            f"{url} | "
            f"Warning: {response.get('message', 'Document already indexed')}\n"
            f"  Vectors found: {response.get('vectors_found', 0):,} (already indexed)\n"
            f"  Hint: {response.get('hint', 'Resubmit with force_delete=true to proceed')}"
        )
        logger.warning(message)
        return
    
    result = response.get("result", {})
    
    # Extract metrics based on response type (estimate or actual indexing)
    is_estimate = "chunks_planned" in result
    
    if is_estimate:
        # Estimate response format
        chunks = result.get("chunks_planned", 0)
        tokens = result.get("tokens_used", 0)
        cost = result.get("embedding_cost", 0)
        
        message = (
            f"{url} | "
            f"Estimated: {chunks:,} chunks | "
            f"Tokens: {tokens:,} | "
            f"Cost: ${cost:.8f}"
        )
    else:
        # Actual indexing response format
        vectors = result.get("vectors_indexed", 0)
        tokens = result.get("tokens_used", 0)
        cost = result.get("embedding_cost", 0)
        
        message = (
            f"{url} | "
            f"Indexed: {vectors:,} vectors | "
            f"Tokens: {tokens:,} | "
            f"Cost: ${cost:.8f}"
        )
    
    # Log with appropriate level
    if status == "success":
        logger.progress(message)
    elif status == "error":
        error_msg = response.get('error', 'Unknown error')
        logger.error(f"{url} | {error_msg}")
    else:
        logger.warning(f"{url} | Unknown status: {status}")

def process_batch(
    input_file: Path,
    output_file: Path,
    api_url: str,
    estimate: bool,
    force_delete: bool,
    max_chunks: Optional[int],
) -> None:
    """Process a batch of URLs from the input file and save results to output file."""
    # Read input file
    try:
        with open(input_file, 'r') as f:
            input_data = json.load(f)
    except Exception as e:
        print(f"Error reading input file {input_file}: {e}")
        sys.exit(1)

    # Prepare request data
    request_data = {
        "items": input_data.get("items", []),
        "estimate": estimate,
        "force_delete": force_delete,
    }
    
    if max_chunks is not None:
        request_data["max_chunks"] = max_chunks

    # Process the batch
    session = create_session()
    
    # Log start of processing
    logger.info("=" * 80)
    logger.info(f"Starting batch processing")
    logger.info(f"Mode: {'ESTIMATE' if estimate else 'PROCESS'}")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"URLs to process: {len(request_data['items'])}")
    logger.info("=" * 80)
    
    try:
        
        # Process each URL individually for better progress tracking
        results = []
        total_chunks = 0
        total_tokens = 0
        total_cost = 0.0
        
        for i, item in enumerate(request_data["items"], 1):
            item_data = {
                "items": [item],
                "estimate": estimate,
                "force_delete": force_delete,
                "max_chunks": request_data.get("max_chunks")
            }
            
            url = item['url']
            logger.info(f"[{i}/{len(request_data['items'])}] Processing {url}")
            
            try:
                # Process the single item
                response = session.post(
                    api_url,
                    json=item_data,
                    headers={"Content-Type": "application/json"},
                    timeout=300  # 5 minutes timeout per item
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract and log the individual result
                if result.get("results"):
                    item_result = result["results"][0]
                    log_response(url, item_result)
                    
                    # Update totals
                    if item_result.get("status") == "success":
                        res = item_result.get("result", {})
                        chunks = res.get("chunks_planned", res.get("chunks_processed", 0))
                        tokens = res.get("tokens_used", 0)
                        cost = res.get("embedding_cost", 0)
                        
                        total_chunks += chunks
                        total_tokens += tokens
                        total_cost += cost
                        
                        logger.debug(f"Processed {url}: {chunks} chunks, {tokens} tokens, ${cost:.6f}")
                
                results.append(item_result)
                
            except Exception as e:
                error_msg = str(e)
                logger.exception(f"Failed to process {item['url']}")
                results.append({
                    "url": item['url'],
                    "doc_type": item['doc_type'],
                    "status": "error",
                    "error": error_msg
                })
        
        # Prepare final result
        final_result = {
            "timestamp": datetime.utcnow().isoformat(),
            "input_file": str(input_file),
            "estimate": estimate,
            "force_delete": force_delete,
            "max_chunks": request_data.get("max_chunks"),
            "results": results,
            "summary": {
                "total_items": len(results),
                "success": sum(1 for r in results if r.get("status") == "success"),
                "errors": sum(1 for r in results if r.get("status") == "error"),
                "total_chunks": total_chunks,
                "total_tokens": total_tokens,
                "total_cost": total_cost
            }
        }
        
        # Save to output file (JSONL format)
        with open(output_file, 'a') as f:
            f.write(json.dumps(final_result) + '\n')
            
            # Add a summary line
            summary_line = {
                "type": "estimate" if estimate else "index",
                "total_chunks": total_chunks,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "successful_items": final_result['summary']['success'],
                "failed_items": final_result['summary']['errors']
            }
            f.write(json.dumps({"summary": summary_line}) + '\n')
        
        # Log final summary
        logger.info("=" * 80)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total items processed: {len(results)}")
        logger.info(f"Successful: {final_result['summary']['success']}")
        logger.info(f"Failed: {final_result['summary']['errors']}")
        logger.info("-" * 40)
        logger.info(f"{'ESTIMATE' if estimate else 'INDEX'} SUMMARY:")
        logger.info(f"  Total chunks: {total_chunks:,}")
        logger.info(f"  Total tokens: {total_tokens:,}")
        logger.info(f"  {'Estimated' if estimate else 'Actual'} cost: ${total_cost:.6f}")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {output_file}")
        logger.info("=" * 80)
        
    except requests.exceptions.RequestException as e:
        logger.critical(f"Error processing batch: {e}", exc_info=True)
        sys.exit(1)

@click.command()
@click.option(
    "--input",
    "input_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Input JSON file with batch items",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
)
@click.option(
    "--api-url",
    default=DEFAULT_API_URL,
    help=f"API endpoint URL (default: {DEFAULT_API_URL})",
)
@click.option(
    "--no-estimate",
    is_flag=True,
    default=False,
    help="Perform actual processing instead of estimation",
)
@click.option(
    "--force-delete",
    is_flag=True,
    default=False,
    help="Force re-indexing of existing documents",
)
@click.option(
    "--max-chunks",
    type=int,
    help="Maximum number of chunks per document",
)
@click.option(
    "--create-sample",
    is_flag=True,
    help="Create a sample input file and exit",
)
def main(
    input_file: Optional[Path],
    output_dir: Path,
    api_url: str,
    no_estimate: bool,
    force_delete: bool,
    max_chunks: Optional[int],
    create_sample: bool,
) -> None:
    """Process documents in batch mode."""
    if create_sample:
        create_sample_input_file()
        return

    if not input_file:
        # Look for input files in the default input directory
        input_files = list(DEFAULT_INPUT_DIR.glob("*.json"))
        if not input_files:
            print(f"No input files found in {DEFAULT_INPUT_DIR}")
            print("Use --create-sample to generate a sample input file")
            sys.exit(1)
        input_file = input_files[0]
        print(f"Using input file: {input_file}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"batch_result_{timestamp}.jsonl"

    process_batch(
        input_file=input_file,
        output_file=output_file,
        api_url=api_url,
        estimate=not no_estimate,
        force_delete=force_delete,
        max_chunks=max_chunks,
    )

def create_sample_input_file() -> None:
    """Create a sample input file with example data."""
    sample_file = DEFAULT_INPUT_DIR / "sample_batch_input.json"
    sample_data = {
        "items": [
            {
                "url": "https://en.wikipedia.org/wiki/Mount_Everest",
                "doc_type": "mediawiki",
                "skip_sections": ["References", "External links", "See also", "Further reading"]
            },
            {
                "url": "https://en.wikipedia.org/wiki/yellowstone_national_park",
                "doc_type": "mediawiki",
                "skip_sections": ["References", "External links", "See also", "Further reading"]
            },
            {
                "url": "https://en.wikipedia.org/wiki/Mont_Blanc",
                "doc_type": "html",
                "skip_sections": ["References", "External links", "See also", "Further reading"],
                "user_agent": "Mozilla/5.0 (compatible; MyBot/1.0; +http://example.com/bot)"
            },
            {
                "url": "https://en.wikipedia.org/wiki/Salcantay",
                "doc_type": "html",
                "skip_sections": ["References", "External links", "See also", "Further reading"],
                "user_agent": "Mozilla/5.0 (compatible; MyBot/1.0; +http://example.com/bot)"
            },
            {
                "url": "https://appalachiantrail.org/wp-content/uploads/2020/07/appalachian-trail-day-hikes-1.pdf",
                "doc_type": "pdf",
                "skip_sections": []
            },
            {
                "url": "file:///app/data/pdf-files-for-upload/<your-file.pdf>",
                "doc_type": "pdf",
                "skip_sections": ["References", "External links", "See also", "Further reading"]
            }
        ],
        "max_chunks": 100,
        "estimate": True,
        "force_delete": False
    }
    
    with open(sample_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Sample input file created: {sample_file}")
    print("You can now run: python -m scripts.batch.process_docs --input " + str(sample_file))

if __name__ == "__main__":
    main()
