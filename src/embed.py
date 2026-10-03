from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np, json

def main():
    data_path = Path(__file__).resolve().parent.parent / "data" 
    with open(data_path / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [c["text"] for c in chunks]
        vectors = model.encode(texts, normalize_embeddings=True)
        np.save(data_path / "embeddings.npy", vectors)

main()