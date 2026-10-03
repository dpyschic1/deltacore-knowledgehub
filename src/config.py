# Embedding model (chunk embeddings, query embeddings, entity resolution)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Local LLM generation (Ollama)
OLLAMA_URL = "http://localhost:11434/api/chat"
GENERATION_MODEL_NAME = "qwen2.5:3b"
TEMPERATURE = 0

# Chunking (ingest.py)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Vector retrieval (retrieve.py / ask.py)
TOP_K = 3
SIMILARITY_THRESHOLD = 0.40

# Graph retrieval (graph_retrieve.py)
MAX_HOPS = 4
