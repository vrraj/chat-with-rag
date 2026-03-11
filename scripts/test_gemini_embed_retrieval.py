import math
import os
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

# Initialize clients
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY environment variable not set")
    exit(1)

genai_client = genai.Client(api_key=api_key)
qdrant_client = QdrantClient(host="localhost", port=6333)

# Configuration
collection_name = "test_gemini_embed_retrieval"
vector_size = 1536
test_text = "What are the main components of a RAG system?"

def print_vector_info(vector, label):
    """Print first 10 values and magnitude of a vector"""
    first_10 = vector[:10]
    magnitude = math.sqrt(sum(x * x for x in vector))
    print(f"\n{label}:")
    print(f"  First 10 values: {[round(x, 6) for x in first_10]}")
    print(f"  Magnitude: {magnitude:.6f}")
    return magnitude

def create_embedding(text, task_type=None):
    """Create embedding with optional task_type"""
    config_params = {"output_dimensionality": vector_size}
    if task_type:
        config_params["task_type"] = task_type
    
    response = genai_client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=[text],
        config=genai.types.EmbedContentConfig(**config_params),
    )
    
    return response.embeddings[0].values

def setup_collection():
    """Setup Qdrant collection"""
    try:
        # Delete existing collection if it exists
        qdrant_client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass
    
    # Create new collection
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"Created collection: {collection_name}")

def main():
    print("=" * 80)
    print("GEMINI EMBEDDING & RETRIEVAL TEST")
    print("=" * 80)
    
    # Setup collection
    setup_collection()
    
    # Step 1: Create embedding with no task_type
    print("\n" + "="*50)
    print("STEP 1: Embedding with NO task_type")
    print("="*50)
    
    embedding_no_task = create_embedding(test_text, task_type=None)
    mag_no_task = print_vector_info(embedding_no_task, "No task_type embedding")
    
    # Save to Qdrant
    point_id = str(uuid.uuid4())
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[PointStruct(
            id=point_id,
            vector=embedding_no_task,
            payload={"text": test_text, "task_type": "none", "step": 1}
        )]
    )
    print(f"Saved to Qdrant with ID: {point_id}")
    
    # Step 2: Create embedding with RETRIEVAL_DOCUMENT task_type
    print("\n" + "="*50)
    print("STEP 2: Embedding with RETRIEVAL_DOCUMENT task_type")
    print("="*50)
    
    embedding_doc = create_embedding(test_text, task_type="RETRIEVAL_DOCUMENT")
    mag_doc = print_vector_info(embedding_doc, "RETRIEVAL_DOCUMENT embedding")
    
    # Save to Qdrant
    point_id = str(uuid.uuid4())
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[PointStruct(
            id=point_id,
            vector=embedding_doc,
            payload={"text": test_text, "task_type": "RETRIEVAL_DOCUMENT", "step": 2}
        )]
    )
    print(f"Saved to Qdrant with ID: {point_id}")
    
    # Compare document embeddings
    diff = max(abs(a - b) for a, b in zip(embedding_no_task, embedding_doc))
    print(f"\nDifference between no_task and RETRIEVAL_DOCUMENT: {diff:.10f}")
    
    # Query tests
    query_text = "How do RAG systems work?"
    
    # Step 3: Query with no task_type
    print("\n" + "="*50)
    print("STEP 3: Query with NO task_type")
    print("="*50)
    
    query_no_task = create_embedding(query_text, task_type=None)
    mag_query_no_task = print_vector_info(query_no_task, "Query (no task_type)")
    
    # Search in Qdrant
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_no_task,
        limit=2,
        with_payload=True
    )
    
    print(f"\nSearch results (query no task_type):")
    for i, result in enumerate(search_results):
        print(f"  Result {i+1}: Score={result.score:.6f}, Task={result.payload['task_type']}, Step={result.payload['step']}")
    
    # Step 4: Query with RETRIEVAL_QUERY task_type
    print("\n" + "="*50)
    print("STEP 4: Query with RETRIEVAL_QUERY task_type")
    print("="*50)
    
    query_search = create_embedding(query_text, task_type="RETRIEVAL_QUERY")
    mag_query_search = print_vector_info(query_search, "Query (RETRIEVAL_QUERY)")
    
    # Search in Qdrant
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_search,
        limit=2,
        with_payload=True
    )
    
    print(f"\nSearch results (query RETRIEVAL_QUERY):")
    for i, result in enumerate(search_results):
        print(f"  Result {i+1}: Score={result.score:.6f}, Task={result.payload['task_type']}, Step={result.payload['step']}")
    
    # Step 5: Query with QUESTION_ANSWERING task_type
    print("\n" + "="*50)
    print("STEP 5: Query with QUESTION_ANSWERING task_type")
    print("="*50)
    
    query_qa = create_embedding(query_text, task_type="QUESTION_ANSWERING")
    mag_query_qa = print_vector_info(query_qa, "Query (QUESTION_ANSWERING)")
    
    # Search in Qdrant
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_qa,
        limit=2,
        with_payload=True
    )
    
    print(f"\nSearch results (query QUESTION_ANSWERING):")
    for i, result in enumerate(search_results):
        print(f"  Result {i+1}: Score={result.score:.6f}, Task={result.payload['task_type']}, Step={result.payload['step']}")
    
    # Compare query embeddings
    print("\n" + "="*50)
    print("QUERY EMBEDDING COMPARISONS")
    print("="*50)
    
    diff_no_search = max(abs(a - b) for a, b in zip(query_no_task, query_search))
    diff_no_qa = max(abs(a - b) for a, b in zip(query_no_task, query_qa))
    diff_search_qa = max(abs(a - b) for a, b in zip(query_search, query_qa))
    
    print(f"Max difference between no_task and RETRIEVAL_QUERY: {diff_no_search:.10f}")
    print(f"Max difference between no_task and QUESTION_ANSWERING: {diff_no_qa:.10f}")
    print(f"Max difference between RETRIEVAL_QUERY and QUESTION_ANSWERING: {diff_search_qa:.10f}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Document embeddings:")
    print(f"  No task_type magnitude: {mag_no_task:.6f}")
    print(f"  RETRIEVAL_DOCUMENT magnitude: {mag_doc:.6f}")
    print(f"\nQuery embeddings:")
    print(f"  No task_type magnitude: {mag_query_no_task:.6f}")
    print(f"  RETRIEVAL_QUERY magnitude: {mag_query_search:.6f}")
    print(f"  QUESTION_ANSWERING magnitude: {mag_query_qa:.6f}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
