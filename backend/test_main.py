import pytest
from httpx import AsyncClient
from backend.main import app
from unittest.mock import patch

@pytest.fixture
async def test_client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_upsert_vector(test_client):
    test_data = {"id": 123, "vector": [0.1, 0.2, 0.3, 0.4]}

    with patch("backend.main.client.upsert") as mock_upsert:
        mock_upsert.return_value = None  # simulate success

        response = await test_client.post("/upsert", json=test_data)

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        mock_upsert.assert_called_once()


# Real Qdrant integration test via FastAPI
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

@pytest.mark.asyncio
async def test_qdrant_fastapi_integration(test_client):
    # Recreate the Qdrant collection before test
    qclient = QdrantClient(host="localhost", port=6333)
    qclient.recreate_collection(
        collection_name="test_vectors",
        vectors_config=VectorParams(size=4, distance=Distance.COSINE)
    )

    # Insert vector via API
    test_data = {"id": 101, "vector": [0.11, 0.22, 0.33, 0.44]}
    response = await test_client.post("/upsert", json=test_data)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    # Query Qdrant directly to confirm
    results = qclient.search(
        collection_name="test_vectors",
        query_vector=[0.11, 0.22, 0.33, 0.44],
        limit=1
    )

    assert len(results) == 1
    assert results[0].id == 101