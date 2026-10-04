import networkx as nx
import json
from pathlib import Path

#convert data to a multi directed graph, so parallel edges can arise
def build_graph(triples):
    G = nx.MultiDiGraph()
    for t in triples:
        G.add_edge(
            t["subject"],
            t["object"],
            relation = t["relation"],
            source_doc_id = t["source_doc_id"],
            source_chunk_index=t["source_chunk_index"],
            raw_subject=t["raw_subject"],
            raw_object=t["raw_object"]
        )
    return G

def main():
    data_path = Path(__file__).resolve().parent.parent.parent / "data"
    with open(data_path / "resolved_triples.json", "r", encoding="utf-8") as r:
        resolved = json.load(r)

    G = build_graph(resolved)

    graph_data = nx.node_link_data(G)

    with open(data_path / "graph.json", "w", encoding="utf-8") as g:
        json.dump(graph_data, g, indent=4)

    print(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

main()