import json
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from pathlib import Path
from config import CHROMA_COLLECTION_NAME, RETRIEVAL_EMBEDDING_MODEL_NAME

#Similar logic as v1, except values get persisted to chromadb instead

def main():
    data_path = Path(__file__).resolve().parent.parent / "data"
    with open(data_path / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    model = SentenceTransformer(RETRIEVAL_EMBEDDING_MODEL_NAME)

    ids = []
    embeddings_input = []
    documents = []
    metadatas = []

    for chunk in chunks:
        chunk_id = f"{chunk["doc_id"]}::{chunk["chunk_index"]}"
        contextualized = chunk["context"] + "\n\n" + chunk["text"]

        ids.append(chunk_id)
        embeddings_input.append(contextualized)
        documents.append(chunk["text"])
        metadatas.append({
            "doc_id": chunk["doc_id"],
            "chunk_index": chunk["chunk_index"],
            "source_path": chunk["source_path"],
            "start_char": chunk["start_char"],
            "end_char": chunk["end_char"],
        })

    vectors = model.encode(embeddings_input, normalize_embeddings=True)

    db_client = PersistentClient(path=str(data_path / "chroma_db"))
    collection = db_client.get_or_create_collection(CHROMA_COLLECTION_NAME)

    #this is database so no overwrite unlike files, hence we need upsert
    collection.upsert(
        ids=ids,
        embeddings=vectors.tolist(),
        documents=documents,
        metadatas=metadatas
    )
    print(f"Indexed {len(ids)} chunks into Chroma collection '{CHROMA_COLLECTION_NAME}'")

main()