import json, requests
from pathlib import Path
from config import OLLAMA_URL, GENERATION_MODEL_NAME, TEMPERATURE

EXTRACTION_SYSTEM_PROMPT = (
    "You are an information extraction system. Read the text and extract "
    "EVERY factual relationship stated in it -- not just the single most "
    "obvious one per entity. For each entity (person, project, technology), "
    "extract ALL of: who leads/owns/built it, what technologies it uses or "
    "depends on, and what purpose or capability it has. Respond with a JSON "
    "object containing a single key \"triples\", whose value is an array of "
    "triples. Each triple has exactly these fields: subject, relation, "
    "object (all plain strings)."
)

EXAMPLE_TEXT = (
    "Project Nova is a billing platform written in Go, using Redis for "
    "caching. It was built by Jamie Lee."
)
EXAMPLE_OUTPUT = (
    '{"triples": ['
    '{"subject": "Project Nova", "relation": "is_a", "object": "billing platform"}, '
    '{"subject": "Project Nova", "relation": "written_in", "object": "Go"}, '
    '{"subject": "Project Nova", "relation": "uses", "object": "Redis"}, '
    '{"subject": "Project Nova", "relation": "built_by", "object": "Jamie Lee"}'
    ']}'
)

def extract_triplets(chunk_text):
    user_prompt = (
        f"Example text: {EXAMPLE_TEXT}\n\n"
        f"Example output: {EXAMPLE_OUTPUT}\n\n"
        f"Now extract triplets from this text:\n{chunk_text}"
    )

    paylod = {
        "model": GENERATION_MODEL_NAME,
        "messages": [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": TEMPERATURE},
    }
    try: 
        response = requests.post(OLLAMA_URL, json=paylod, timeout=360)
    except requests.exceptions.ReadTimeout or requests.exceptions.ConnectionError:
        print(f"Warning: request to ollama failed for chunk, skipping: {chunk_text[:100]!r}")
        return []

    response.raise_for_status()
    
    content = response.json()["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        print(f"WARNING: invalid JSON for chunk, skipping: {content[:100]!r}")
        return []

    if not isinstance(parsed, dict) or not isinstance(parsed.get("triples"), list):
        print(f"WARNING: expected an object with a 'triples' array, got: {content[:100]!r}")
        return []

    triples_list = parsed["triples"]

    valid_triplets = []
    for item in triples_list:
        if isinstance(item, dict) and {"subject", "relation", "object"} <= item.keys():
            valid_triplets.append(item)
        else:
            print(f"WARNING: malformed triple, skipping: {item!r}")

    return valid_triplets

def main():
    data_path = Path(__file__).resolve().parent.parent / "data"
    with open(data_path / "chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)

    all_triples = []
    for chunk in chunks:
        print(f"Extracting chunk doc id:{chunk['doc_id']}  chunk_index: {chunk['chunk_index']}")
        triplets = extract_triplets(chunk["text"])
        for t in triplets:
            t["source_doc_id"] = chunk["doc_id"]
            t["source_chunk_index"] = chunk["chunk_index"]
            all_triples.append(t)

    with open(data_path / "triples.json", "w", encoding="utf-8") as t:
        json.dump(all_triples, t, indent=4, ensure_ascii=False)

main()