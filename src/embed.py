from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np, json
from config import EMBEDDING_MODEL_NAME

def main():
    data_path = Path(__file__).resolve().parent.parent / "data"
    with open(data_path / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        texts = [c["text"] for c in chunks]
        vectors = model.encode(texts, normalize_embeddings=True)
        np.save(data_path / "embeddings.npy", vectors)

main()