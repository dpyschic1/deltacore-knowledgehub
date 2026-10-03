import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter, defaultdict
from config import EMBEDDING_MODEL_NAME

def get_unique_entities(triples):
    names = set()
    for t in triples:
        names.add(t["subject"])
        names.add(t["object"])
    return sorted(names)

def cluster_entities(names, model, threshold):
    vectors = model.encode(names, normalize_embeddings=True)
    similarities = np.dot(vectors, vectors.T)

    #Union-find to consolidate graph entities for transitive relationships A->B->C
    parent = {name: name for name in names}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(names)):
        for j in range(i+1, len(names)):
            if(similarities[i][j]) >= threshold: #Consolidating only when vec similarities cross the threshold
                union(names[i], names[j])

    return parent

#these would be names that occur the most often, so parent node has the highest number of cild nodes
#picking the most common names and updating clusters to use those as roots
def pick_canonical_names(names, parent, triples):

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    counts = Counter()
    for t in triples:
        counts[t["subject"]] += 1
        counts[t["object"]] += 1
    clusters = defaultdict(list)
    for name in names:
        root = find(name)
        clusters[root].append(name)

    canonical = {}
    for root, members in clusters.items():
        representative = max(members, key=lambda name: (counts[name], -len(name)))
        for name in members:
            canonical[name] = representative

    return canonical

def main():
    data_path = Path(__file__).resolve().parent.parent / "data" 
    with open(data_path / "triples.json", "r", encoding="utf-8") as c:
        triples = json.load(c)

    names = get_unique_entities(triples)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    parent = cluster_entities(names, model, threshold=0.63)
    canonical = pick_canonical_names(names, parent, triples)

    resolved = []
    for t in triples:
        resolved.append({
            "subject": canonical[t["subject"]],
            "relation": t["relation"],
            "object": canonical[t["object"]],
            "raw_subject": t["subject"],
            "raw_object": t["object"],
            "source_doc_id": t["source_doc_id"],
            "source_chunk_index": t["source_chunk_index"]
        })

    with open(data_path/ "entity_map.json", "w", encoding="utf-8") as e:
        json.dump(canonical, e, indent=4)

    with open(data_path / "resolved_triples.json", "w", encoding="utf-8") as r:
        json.dump(resolved, r, indent=4)

main()