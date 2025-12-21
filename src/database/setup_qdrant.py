import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv

load_dotenv()

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_qdrant_client():
    """
    Returns a QdrantClient instance based on environment variables.
    Defaults to in-memory mode if no URL is provided.
    """
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if qdrant_url:
        logger.info(f"Connecting to Qdrant at {qdrant_url}")
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    # Fallback to local disk storage for persistence, or memory
    logger.info("QDRANT_URL not set. Using local storage in 'data/qdrant_storage'.")
    return QdrantClient(path="data/qdrant_storage")

def setup_collection(collection_name: str, vector_size: int = 1536):
    """
    Creates a Qdrant collection if it doesn't already exist.
    
    Args:
        collection_name (str): Name of the collection.
        vector_size (int): Dimension of vectors (default 1536 for OpenAI text-embedding-3-small/large or ada-002).
    """
    client = get_qdrant_client()
    
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)
    
    if not exists:
        logger.info(f"Creating collection '{collection_name}' with vector size {vector_size}...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info(f"Collection '{collection_name}' created successfully.")
    else:
        logger.info(f"Collection '{collection_name}' already exists.")

if __name__ == "__main__":
    # Example setup
    COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "job_market")
    setup_collection(COLLECTION_NAME)
