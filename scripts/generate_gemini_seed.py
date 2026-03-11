#!/usr/bin/env python3
"""
Generate Gemini embeddings from existing OpenAI seed data.

This script reads the existing docs-index-seed.jsonl file (which has OpenAI embeddings),
extracts the text content from the payloads, generates new embeddings using Gemini,
and creates a new seed file docs-index-seed-gemini.jsonl.

Usage:
  python scripts/generate_gemini_seed.py
  python scripts/generate_gemini_seed.py --input data/docs-index-seed.jsonl --output data/docs-index-seed-gemini.jsonl

Requirements:
  - GEMINI_API_KEY must be set in environment
  - Existing seed file with text content in payloads
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any, Iterable
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

# Make the project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.llm.llm_client import embed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default file paths
DEFAULT_INPUT = "data/docs-index-seed.jsonl"
DEFAULT_OUTPUT = "data/docs-index-seed-gemini.jsonl"


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Iterate over JSONL file and yield parsed objects."""
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")
                continue


def extract_text_from_payload(payload: Dict[str, Any]) -> str:
    """Extract text content from the payload."""
    # Try different field names that might contain the text
    text_fields = ["text", "content", "document_text", "chunk_text"]
    
    for field in text_fields:
        if field in payload and isinstance(payload[field], str):
            return payload[field]
    
    # If no standard field, try to find the first string value
    for key, value in payload.items():
        if isinstance(value, str) and len(value) > 50:  # Assume longer strings are content
            return value
    
    logger.warning(f"Could not extract text from payload: {payload}")
    return ""


def generate_gemini_embedding(text: str, max_retries: int = 3) -> List[float]:
    """Generate Gemini embedding for the given text."""
    if not text.strip():
        raise ValueError("Empty text provided for embedding")
    
    for attempt in range(max_retries):
        try:
            resp = embed(
                model_key="gemini:native-embed",
                texts=text,
                dimensions=None  # Let Gemini determine dimensions (usually 768 or 1536)
            )
            
            if not resp or not hasattr(resp, 'data') or not resp.data:
                raise ValueError("Invalid response from embedding service")
            
            embedding = resp.data[0].embedding
            return embedding
            
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Embedding attempt {attempt + 1} failed: {e}. Retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff
    
    raise RuntimeError("Failed to generate embedding after all retries")


def main() -> None:
    """Main function to generate Gemini seed data."""
    parser = argparse.ArgumentParser(description="Generate Gemini embeddings from existing seed data")
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Input JSONL file (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output JSONL file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--batch-size", type=int, default=10, help="Processing batch size (default: 10)")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples to process (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Only extract text, don't generate embeddings")
    args = parser.parse_args()

    # Check environment
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY is not set. Please set it in your environment or .env file.")
        sys.exit(1)

    # Check input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    logger.info(f"Reading seed data from: {args.input}")
    logger.info(f"Output will be written to: {args.output}")
    
    if args.dry_run:
        logger.info("DRY RUN: Only extracting text, not generating embeddings")

    # Process records
    processed = 0
    skipped = 0
    batch = []
    
    with open(args.output, "w", encoding="utf-8") as out_file:
        for i, record in enumerate(iter_jsonl(args.input)):
            if args.max_samples and i >= args.max_samples:
                break
            
            # Extract text from payload
            payload = record.get("payload", {})
            text = extract_text_from_payload(payload)
            
            if not text:
                skipped += 1
                logger.warning(f"Skipping record {i+1}: no text content found")
                continue
            
            # Generate new record
            new_record = {
                "id": record["id"],
                "payload": payload  # Keep the same payload
            }
            
            if args.dry_run:
                # For dry run, just use a placeholder vector
                new_record["vector"] = [0.0] * 1536  # Placeholder
            else:
                try:
                    # Generate Gemini embedding
                    embedding = generate_gemini_embedding(text)
                    new_record["vector"] = embedding
                    logger.info(f"Processed record {i+1}: embedding dimension {len(embedding)}")
                except Exception as e:
                    logger.error(f"Failed to generate embedding for record {i+1}: {e}")
                    skipped += 1
                    continue
            
            batch.append(new_record)
            processed += 1
            
            # Write batch
            if len(batch) >= args.batch_size:
                for record in batch:
                    out_file.write(json.dumps(record) + "\n")
                out_file.flush()
                logger.info(f"Wrote {len(batch)} records (total: {processed})")
                batch.clear()
        
        # Write remaining records
        if batch:
            for record in batch:
                out_file.write(json.dumps(record) + "\n")
            logger.info(f"Wrote final batch of {len(batch)} records")

    logger.info(f"Processing complete!")
    logger.info(f"Total processed: {processed}")
    logger.info(f"Total skipped: {skipped}")
    logger.info(f"Output file: {args.output}")
    
    if args.dry_run:
        logger.info("DRY RUN completed. Run without --dry-run to generate actual embeddings.")


if __name__ == "__main__":
    main()
