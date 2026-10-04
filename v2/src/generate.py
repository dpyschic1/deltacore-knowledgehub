
import requests

from config import GENERATION_MODEL_NAME, OLLAMA_URL, TEMPERATURE
from retrieve import retrieve_chunks


SYSTEM_PROMPT = """You are a question-answering assistant. Answer strictly and only using the numbered sources provided below.

First, reason step by step: for each fact relevant to the question, identify which numbered source states it, and quote or closely paraphrase that source's actual words before relying on it. Do not introduce any fact, name, team, or relationship that is not explicitly stated in one of the numbered sources, even if it seems plausible.

If the question asks "besides X" or "excluding X", your reasoning must explicitly check and rule out X before considering any other candidate, and your final answer must NOT be X even if X appears prominently in the sources.

After your reasoning, give your final answer on its own line starting with "Final answer:", followed by the actual answer itself (a name or fact -- never leave this blank or citations-only), with the matching source number(s) in square brackets immediately after it, like "Final answer: Dennis Rodman [3]". Only cite a source number if you just quoted or paraphrased that exact source during your reasoning -- never attach a citation to a claim you did not verify against the source text.

If the sources do not contain enough information to answer, write "Final answer: I don't know based on the provided sources." and nothing else. Do not use outside knowledge."""

def build_source_block(retrieved_chunks):
    citation_map = {}
    lines = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        citation_map[i] = chunk
        lines.append(f"[{i}] (source: {chunk['doc_id']} chunk {chunk['chunk_index']})\n{chunk['text']}")
    return "\n\n".join(lines), citation_map

def generate_answer(question, retrieved_chunks):
    source_blocks, citation_map = build_source_block(retrieved_chunks)
    user_prompt = f"Sources:\n{source_blocks}\n\nQuestion:{question}"
    payload = {
                "model": GENERATION_MODEL_NAME,
                        "messages": [
                            {
                                "role": "system",
                                "content": SYSTEM_PROMPT
                            },
                            {
                                "role": "user", 
                                "content": user_prompt
                            },
                        ],
                        "stream": False,
                        "options": {
                            "temperature": TEMPERATURE
                        }
            }
    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()
    answer_text = data["message"]["content"]
    return answer_text, citation_map, source_blocks
