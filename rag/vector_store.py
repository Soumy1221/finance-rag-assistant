from functools import lru_cache
import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_DIR, COLLECTION_NAME

@lru_cache(maxsize=1)
def embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_collection():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

def add_chunks(chunks):
    if not chunks:
        return 0
    collection = get_collection()
    texts = [x["text"] for x in chunks]
    vectors = embedding_model().encode(texts, normalize_embeddings=True).tolist()
    ids = [f'{x["metadata"]["source"]}::{x["metadata"]["chunk"]}' for x in chunks]
    collection.upsert(ids=ids, documents=texts, embeddings=vectors,
                      metadatas=[x["metadata"] for x in chunks])
    return len(chunks)

def retrieve(query, k=4):
    collection = get_collection()
    if collection.count() == 0:
        return []
    vector = embedding_model().encode([query], normalize_embeddings=True).tolist()[0]
    result = collection.query(query_embeddings=[vector], n_results=min(k, collection.count()),
                              include=["documents", "metadatas", "distances"])
    return [
        {"text": d, "metadata": m, "distance": float(dist)}
        for d, m, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]
