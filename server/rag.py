import os

from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from typing import List

dense_model = "sentence-transformers/all-MiniLM-L6-v2"
sparse_model = "Qdrant/bm25"
collection_name = f"gauntlet-bot"

client = None
def init():
    global client
    if client:
        return
    
    # Local
    # client = QdrantClient(path="qdrant_db")
    # client = QdrantClient(":memory:")

    # Cloud
    url="https://873902b1-f25a-4641-86e3-feff2cf9010b.us-east4-0.gcp.cloud.qdrant.io:6333"
    client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))

    client.set_model(dense_model)
    client.set_sparse_model(sparse_model)
        
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name, 
            vectors_config=client.get_fastembed_vector_params(),
            sparse_vectors_config=client.get_fastembed_sparse_vector_params(), 
        )
init()

def add_documents(docs):
    client.add(
        collection_name=collection_name,
        documents=docs,
        parallel=8,
    )

def similarity_search(query: str, uid: str | None = None) -> List[str]:
    filter = Filter(must=[
        FieldCondition(key="metadata.uid", match=MatchValue(value=uid)),
    ]) if uid else None

    docs = client.query(
        collection_name=collection_name,
        query_text=query,
        query_filter=filter,
        limit=4,
    )
    return [doc.document for doc in docs]