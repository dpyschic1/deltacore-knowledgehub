import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np, json
from config import TOP_K, SIMILARITY_THRESHOLD, RETRIEVAL_EMBEDDING_MODEL_NAME, RETRIEVAL_QUERY_PREFIX

def main():
    data_path = Path(__file__).resolve().parent.parent / "data" 
    chunks = []
    vectors = []
    with open(data_path / "chunks.json", "r", encoding="utf-8") as c:
        chunks = json.load(c)
    vectors = np.load(data_path / "embeddings.npy")

    query = sys.argv[1]

    model = SentenceTransformer(RETRIEVAL_EMBEDDING_MODEL_NAME)
    query_vector = model.encode([RETRIEVAL_QUERY_PREFIX + query], normalize_embeddings=True)[0]

    similarities = np.dot(vectors, query_vector)
    
    top_indices = np.argsort(similarities)[::-1][:TOP_K]

    for idx in top_indices:
        chunk = chunks[idx]
        print (similarities[idx], chunk["doc_id"], chunk["chunk_index"], chunk["text"])

def retrieve_chunks(question, top_k):
    data_path = Path(__file__).resolve().parent.parent / "data" 
    chunks = []
    vectors = []
    with open(data_path / "chunks.json", "r", encoding="utf-8") as c:
        chunks = json.load(c)
    vectors = np.load(data_path / "embeddings.npy")
    model = SentenceTransformer(RETRIEVAL_EMBEDDING_MODEL_NAME)
    query_vector = model.encode([RETRIEVAL_QUERY_PREFIX + question], normalize_embeddings=True)[0]

    similarities = np.dot(vectors, query_vector)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    result = []
    for idx in top_indices:
        score = similarities[idx]
        if score < SIMILARITY_THRESHOLD:
            continue

        chunk = chunks[idx]
        result.append({
            "doc_id": chunk["doc_id"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "score": f"{int(score * 100)}%",
            "origin": "vector"
        })

    return result

