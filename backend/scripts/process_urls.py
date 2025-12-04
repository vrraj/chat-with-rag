#!/usr/bin/env python3
"""
Script for batch processing URLs with configurable parameters.

Example usage:
    # Process URLs from file
    python -m backend.scripts.process_urls --input urls.jsonl --output results.jsonl
    
    # Process single URL
    python -m backend.scripts.process_urls --url https://example.com --action index --type html
"""
import json
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import List, Optional, TextIO, Dict, Any
from urllib.parse import urlparse

import httpx
from pydantic import ValidationError

# Add project root to path if needed
sys.path.append(str(Path(__file__).parent.parent))

from backend.scripts.schemas import URLItem, ProcessingResult, ContentType, ActionType
from backend.core.config import HTMLConfig, PDFConfig, MediaWikiConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class URLProcessor:
    """Process URLs with configurable parameters."""
    
    def __init__(self, default_config: Optional[Dict[str, Any]] = None):
        """Initialize with optional default configuration."""
        self.default_config = default_config or {}
    
    async def process_url(self, url_item: URLItem) -> ProcessingResult:
        """Process a single URL with the given configuration."""
        start_time = time.time()
        result = {
            "url": str(url_item.url),
            "action": url_item.action,
            "config_used": {},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
        }
        
        try:
            # Process based on content type
            config = self._get_config(url_item)
            result["config_used"] = config.dict()
            
            if url_item.action == ActionType.ESTIMATE:
                # Simulate estimation
                result["result"] = {"estimated_chunks": 10, "estimated_size": "1.2MB"}
                result["status"] = "success"
            else:  # INDEX
                # TODO: Implement actual indexing
                result["result"] = {"chunks_processed": 10, "status": "indexed"}
                result["status"] = "success"
                
        except Exception as e:
            logger.error(f"Error processing {url_item.url}: {str(e)}", exc_info=True)
            result["status"] = "error"
            result["error"] = str(e)
        
        result["duration_seconds"] = time.time() - start_time
        return ProcessingResult(**result)
    
    def _get_config(self, url_item: URLItem):
        """Get the appropriate config object based on URL and config."""
        config_data = (url_item.config or {}).dict() if url_item.config else {}
        
        # Determine content type from URL if not specified
        if not config_data.get('type'):
            if str(url_item.url).endswith('.pdf'):
                config_data['type'] = ContentType.PDF
            else:
                config_data['type'] = ContentType.HTML
        
        # Create appropriate config object
        config_class = {
            ContentType.HTML: HTMLConfig,
            ContentType.PDF: PDFConfig,
            ContentType.MEDIAWIKI: MediaWikiConfig
        }.get(config_data['type'], HTMLConfig)
        
        return config_class(**{**self.default_config, **config_data})

async def process_urls(
    input_file: Optional[TextIO] = None,
    output_file: Optional[TextIO] = None,
    url: Optional[str] = None,
    **kwargs
) -> None:
    """Process URLs from file or single URL."""
    processor = URLProcessor()
    
    if url:
        # Process single URL
        url_item = URLItem(url=url, **kwargs)
        result = await processor.process_url(url_item)
        output = result.json(indent=2)
        if output_file:
            output_file.write(output + "\n")
        print(output)
    elif input_file:
        # Process URLs from file
        for line in input_file:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                url_item = URLItem(**data)
                result = await processor.process_url(url_item)
                output = result.json()
                if output_file:
                    output_file.write(output + "\n")
                print(output)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {line.strip()}")
            except ValidationError as e:
                logger.error(f"Invalid URL item: {e}")
    else:
        raise ValueError("Either --input or --url must be specified")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process URLs with configurable parameters')
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input',
        type=argparse.FileType('r'),
        help='Input file with URLs (one per line or JSONL)'
    )
    input_group.add_argument(
        '--url',
        type=str,
        help='Single URL to process'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=argparse.FileType('w'),
        help='Output file for results (default: stdout)'
    )
    
    # Processing options
    parser.add_argument(
        '--action',
        type=str,
        choices=['estimate', 'index'],
        default='estimate',
        help='Action to perform'
    )
    parser.add_argument(
        '--type',
        type=str,
        choices=['html', 'pdf', 'mediawiki'],
        help='Content type (default: auto-detect from URL)'
    )
    parser.add_argument(
        '--max-chunks',
        type=int,
        default=0,
        help='Maximum number of chunks to process (0 for no limit)'
    )
    parser.add_argument(
        '--force-delete',
        action='store_true',
        help='Force delete existing content'
    )
    
    return parser.parse_args()

async def main():
    """Main entry point."""
    args = parse_args()
    
    # Prepare kwargs for process_urls
    kwargs = {
        'input_file': args.input,
        'output_file': args.output,
        'url': args.url,
        'action': args.action,
    }
    
    # Add config if any config options are provided
    config = {}
    if args.type:
        config['type'] = args.type
    if args.max_chunks is not None:
        config['max_chunks'] = args.max_chunks
    if args.force_delete is not None:
        config['force_delete'] = args.force_delete
    
    if config:
        kwargs['config'] = config
    
    try:
        await process_urls(**kwargs)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
