import sys
import json
import networkx as nx
from pathlib import Path
from retrieve import retrieve_chunks
from custom_graphrag.graph_retrieve import graph_retrieve
from generate import generate_answer
from config import TOP_K

def load_graph():
    data_path = Path(__file__).resolve().parent.parent / "data"
    data = json.load(open(data_path / "graph.json", encoding="utf-8"))
    return nx.node_link_graph(data)

def main():
    question = sys.argv[1]

    G = load_graph()

    vector_results = retrieve_chunks(question, TOP_K)
    graph_results = graph_retrieve(question, G)
    retrieved = vector_results + graph_results
    answer = generate_answer(question, retrieved)

    print("Answer:")
    print(answer)
    print()
    print("Sources")
    for r in retrieved:
        if r["origin"] == "vector":
            print(f"- {r['doc_id']} (chunk {r['chunk_index']}, score {r['score']})")
        else:
            print(f"- {r['doc_id']} (chunk {r['chunk_index']}, graph fact: \"{r['text']}\")")
main()