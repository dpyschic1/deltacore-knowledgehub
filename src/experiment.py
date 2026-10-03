import json
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

model = SentenceTransformer(EMBEDDING_MODEL_NAME)
names = ["PostgreSQL", "Kubernetes", "Vector databases", "Terraform", "Priya Patel", "Sofia Ramos"]
vecs = model.encode(names, normalize_embeddings=True)
for i in range(len(names)):
    for j in range(i+1, len(names)):
        print(names[i], "<->", names[j], "=", vecs[i] @ vecs[j])