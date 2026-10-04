import requests
from config import GENERATION_MODEL_NAME, ROUTER_SYSTEM_PROMPT, TEMPERATURE, OLLAMA_URL, SIMPLE_TOP_K, SIMPLE_RERANK_TOP_N, COMPLEX_TOP_K

def classify_query(question):
    payload = {
            "model": GENERATION_MODEL_NAME,
                    "messages": [
                        {
                            "role": "system",
                            "content": ROUTER_SYSTEM_PROMPT
                        },
                        {
                            "role": "user", 
                            "content": question
                        },
                    ],
                    "stream": False,
                    "options": {
                        "temperature": TEMPERATURE
                    }
        }
    response = requests.post(OLLAMA_URL, json=payload, timeout=100)
    response.raise_for_status()
    data = response.json()
    content = data["message"]["content"]
    label = content.strip().lower()

    if "simple" in label and "complex" not in label:
        return {"label": "SIMPLE", "top_k": SIMPLE_TOP_K, "rerank_top_k": SIMPLE_RERANK_TOP_N}
    else:
        return {"label": "COMPLEX", "top_k": COMPLEX_TOP_K, "rerank_top_k": None}
