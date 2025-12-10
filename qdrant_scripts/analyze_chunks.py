#!/usr/bin/env python3
"""
Analyze chunks for a specific base URL in Qdrant.

Example usage:
    python qdrant_scripts/analyze_chunks.py --base-url "https://en.wikipedia.org/wiki/mount_everest"
"""
import argparse
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from collections import defaultdict

# Import scroll_points from qdrant_ops
from qdrant_ops import scroll_points, DEFAULT_HOST, DEFAULT_PORT, DEFAULT_COLLECTION

def analyze_chunks(host, port, collection, base_url):
    client = QdrantClient(host=host, port=port)
    base_url_lower = base_url.lower()
    
    print(f"Analyzing chunks for: {base_url}")
    print(f"Using collection: {collection} on {host}:{port}")
    
    # Get all points for this base_url
    filter_condition = Filter(
        must=[FieldCondition(key="base_url_lower", match=MatchValue(value=base_url_lower))]
    )
    
    # Get all points with their payload
    print("\nFetching points from Qdrant...")
    points = []
    for point in scroll_points(client, collection, flt=filter_condition, with_payload=True, with_vectors=False):
        points.append(point)
    
    print(f"\nTotal points found: {len(points)}")
    
    if not points:
        print("No points found for the given base URL.")
        return
    
    # Analyze payload fields
    payload_fields = set()
    for point in points:
        payload_fields.update(point.payload.keys())
    
    print("\nPayload fields found:")
    for field in sorted(payload_fields):
        print(f"- {field}")
    
    # Group by document_id if it exists, otherwise group by base_url
    doc_groups = defaultdict(list)
    for point in points:
        doc_id = point.payload.get('document_id')
        if doc_id is None:
            doc_id = point.payload.get('base_url', 'no_document_id')
        doc_groups[doc_id].append(point)
    
    print(f"\nFound {len(doc_groups)} unique document groups")
    
    total_chunks = 0
    # Show statistics for each document group
    for i, (doc_id, doc_points) in enumerate(doc_groups.items(), 1):
        print(f"\n=== Document Group {i} ===")
        print(f"Document ID: {doc_id}")
        print(f"Number of chunks: {len(doc_points)}")
        total_chunks += len(doc_points)
        
        # Show unique values for some important fields
        fields_to_check = ['chunk_index', 'total_chunks', 'title', 'source', 'updated_at', 'chunk_id', 'section', 'subsection']
        for field in fields_to_check:
            if field in payload_fields:
                unique_values = {str(p.payload.get(field)) for p in doc_points}
                if len(unique_values) == 1:
                    print(f"  {field}: {next(iter(unique_values))}")
                else:
                    print(f"  {field} has {len(unique_values)} unique values")
                    if field in ['chunk_index', 'total_chunks'] and all(v is not None and v != 'None' for v in unique_values):
                        try:
                            print(f"    Range: {min(int(float(v)) for v in unique_values if v is not None and v != 'None')}-{max(int(float(v)) for v in unique_values if v is not None and v != 'None')}")
                        except (ValueError, TypeError):
                            pass
        
        # Show sample of chunk content
        sample = next((p for p in doc_points if 'text' in p.payload), None)
        if sample:
            text = sample.payload['text']
            print(f"\n  Sample text: {text[:200]}...")
    
    print(f"\n=== Summary ===")
    print(f"Total document groups: {len(doc_groups)}")
    print(f"Total chunks across all groups: {total_chunks}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze chunks for a specific base URL in Qdrant")
    parser.add_argument("--base-url", type=str, required=True, 
                       help="Base URL to analyze (case-insensitive)")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, 
                       help=f"Qdrant host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, 
                       help=f"Qdrant port (default: {DEFAULT_PORT})")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION, 
                       help=f"Collection name (default: {DEFAULT_COLLECTION})")
    
    args = parser.parse_args()
    
    analyze_chunks(
        host=args.host,
        port=args.port,
        collection=args.collection,
        base_url=args.base_url
    )
