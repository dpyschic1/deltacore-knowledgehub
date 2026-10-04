import json
import bm25s
from pathlib import Path
from chromadb import PersistentClient
from sentence_transformers import CrossEncoder, SentenceTransformer
from config import CHROMA_COLLECTION_NAME, RERANK_MODEL_NAME, RETRIEVAL_EMBEDDING_MODEL_NAME, RETRIEVAL_QUERY_PREFIX
from classify import classify_query

data_path = Path(__file__).resolve().parent.parent / "data"
with open(data_path / "chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
chunk_ids = [f"{c['doc_id']}::{c['chunk_index']}" for c in chunks]
chunk_lookup = {chunk_id: chunk for chunk_id, chunk in zip(chunk_ids, chunks)}
contextualized_texts = [c["context"] + "\n\n" + c["text"] for c in chunks]
corpus_tokens = bm25s.tokenize(contextualized_texts)
bm25_index = bm25s.BM25()
bm25_index.index(corpus_tokens)

db_client = PersistentClient(path=str(data_path / "chroma_db"))
collection = db_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
model = SentenceTransformer(RETRIEVAL_EMBEDDING_MODEL_NAME)
cross_encoder = CrossEncoder(RERANK_MODEL_NAME)

def vector_search(query, top_k):
    top_k = min(top_k, len(chunk_ids))
    vector = model.encode(RETRIEVAL_QUERY_PREFIX + query, normalize_embeddings=True)
    results = collection.query(query_embeddings=[vector], n_results=top_k)
    return results["ids"][0]

def bm25_search(query, top_k):
    top_k = min(top_k, len(chunk_ids))
    query_tokens = bm25s.tokenize(query, stopwords="english")
    result = bm25_index.retrieve(query_tokens, corpus=chunk_ids, k=top_k)
    return result.documents[0]

def rrf_fuse(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for position, chunk_id in enumerate(ranked_list):
            rank = position + 1
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)

def rerank(query, chunk_ids, top_n):
    pairs = [(query, chunk_lookup[cid]["text"]) for cid in chunk_ids]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(chunk_ids, scores), key=lambda x: x[1], reverse=True)
    return [cid for cid, score in ranked[:top_n]]

def retrieve_chunks(query):
    route = classify_query(query)
    vec_ids = vector_search(query, route["top_k"])
    bm25_ids = bm25_search(query, route["top_k"])
    fused_ids = rrf_fuse([vec_ids, bm25_ids])

    if route["rerank_top_k"] is not None:
        final_ids = rerank(query, fused_ids, route["rerank_top_k"])
    else:
        final_ids = fused_ids
    return [chunk_lookup[cid] for cid in final_ids]
