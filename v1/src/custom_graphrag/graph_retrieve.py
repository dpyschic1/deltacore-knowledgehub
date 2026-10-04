import networkx as nx
from config import MAX_HOPS

def find_matching_entities(question, node_names):
    q = question.lower()
    return [name for name in node_names if name.lower() in q]

def graph_retrieve(question, G, max_hops=MAX_HOPS):
    start_entities = find_matching_entities(question, list(G.nodes()))
    if not start_entities:
        return []

    #converting to undirected here so both forward and reverse relationships can be retrieved
    # we could have for example A->B->C and also B->C->D->A
    undirected = G.to_undirected()
    nodes_in_range = set()
    for entity in start_entities:
        ego = nx.ego_graph(undirected, entity, radius=max_hops)
        nodes_in_range.update(ego.nodes())

    results = []
    seen = set()
    for u, v, key, data in G.edges(keys=True, data=True):
        if u in nodes_in_range and v in nodes_in_range:
            edge_id = (u, v, key)
            if edge_id not in seen:
                seen.add(edge_id)
                results.append({
                    "doc_id": data["source_doc_id"],
                    "chunk_index": data["source_chunk_index"],
                    "text": f"{u} {data['relation']} {v}",
                    "origin": "graph"
                })
    return results

