import os

from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, RetrievalMode, QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, FieldCondition, Filter, MatchValue, SparseVectorParams, VectorParams
from typing import List

client = None
def init():
    global client
    if client:
        return
    
    # Local
    # client = QdrantClient(path="qdrant_db")

    # Cloud
    url="https://873902b1-f25a-4641-86e3-feff2cf9010b.us-east4-0.gcp.cloud.qdrant.io:6333"
    client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))
        
    if not client.collection_exists(collection_name="my_collection"):
        client.create_collection(
            collection_name="my_collection", 
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            sparse_vectors_config={"langchain-sparse": SparseVectorParams()},
        )
init()

embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name="us-east-2")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

vector_store = QdrantVectorStore(
    client=client,
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    collection_name="my_collection",

    # Uses a hybrid of semantic search and keyword search for more accuracy
    retrieval_mode=RetrievalMode.HYBRID,
)

def add_documents(docs):
    if type(docs[0]) == str:
        docs = [Document(page_content=doc) for doc in docs]
    return vector_store.add_documents(docs)

def similarity_search(query: str, uid: str | None = None) -> List[str]:
    filter = Filter(must=[
        FieldCondition(key="metadata.uid", match=MatchValue(value=uid)),
    ]) if uid else None
    docs = vector_store.similarity_search(query, filter=filter)
    return [doc.page_content for doc in docs]