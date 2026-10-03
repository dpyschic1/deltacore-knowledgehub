import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:3b"

SYSTEM_PROMT = (
    "You are a document question-answering assistant. Answer strictly and "
    "only using the text inside the Context section below. Do not use any "
    "outside knowledge, even if you are confident it is correct. If the "
    "context does not explicitly contain the answer, respond with exactly: "
    "\"I don't know based on the provided documents.\" Do not guess or infer "
    "beyond what the context states."
)
TEMPERATURE = 0
def build_user_prompt(question, retrieved_chunks):
    blocks = []
    for chunk in retrieved_chunks:
        blocks.append(f"[source: {chunk["doc_id"]} chunk {chunk["chunk_index"]}]\n{chunk["text"]}")

    context_text = "\n".join(blocks)
    return f"Context:\n{context_text}\n\nQuestion: {question}"

def generate_answer(question, retrieved_chunks):
    user_prompt = build_user_prompt(question, retrieved_chunks)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMT},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {
            "temperature": TEMPERATURE
        }
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]
