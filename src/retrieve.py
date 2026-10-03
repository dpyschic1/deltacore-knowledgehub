import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np, json

TOP_K = 3
SIMILARITY_THRESHOLD = 0.40
def main():
    data_path = Path(__file__).resolve().parent.parent / "data" 
    chunks = []
    vectors = []
    with open(data_path / "chunks.json", "r", encoding="utf-8") as c:
        chunks = json.load(c)
    vectors = np.load(data_path / "embeddings.npy")

    query = sys.argv[1]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = model.encode([query], normalize_embeddings=True)[0]

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
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = model.encode([question], normalize_embeddings=True)[0]

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

