from config import ROUTER_SYSTEM_PROMPT, SIMPLE_TOP_K, SIMPLE_RERANK_TOP_N, COMPLEX_TOP_K
from llm_client import chat

def classify_query(question):
    content = chat("classify", ROUTER_SYSTEM_PROMPT, question)
    label = content.strip().lower()

    if "simple" in label and "complex" not in label:
        return {"label": "SIMPLE", "top_k": SIMPLE_TOP_K, "rerank_top_k": SIMPLE_RERANK_TOP_N}
    else:
        return {"label": "COMPLEX", "top_k": COMPLEX_TOP_K, "rerank_top_k": None}
