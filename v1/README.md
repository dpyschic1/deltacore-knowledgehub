# PythonBasedRag

A local, from-scratch RAG (and GraphRAG) pipeline — document ingestion, chunking, embeddings, vector retrieval, entity/relationship extraction, and graph-based retrieval, all running against a local LLM via Ollama. No GPU required.

## Prerequisites

- Python 3.11+ (developed against 3.14; if you hit package install issues on a very new Python version, fall back to 3.11/3.12)
- [Ollama](https://ollama.com) installed and running
- The following models pulled locally:
  ```
  ollama pull qwen2.5:3b
  ```

## Setup

```powershell
python -m venv ragExpirement
ragExpirement\Scripts\activate
pip install -r requirements.txt
```

## Project layout

```
docs/   source documents (.md / .txt) to index
src/    pipeline scripts
data/   generated artifacts (chunks, embeddings, triples, graph) — all reproducible from docs/
```

## Pipeline

Run in order from `src/` whenever `docs/` changes:

1. `python ingest.py` — chunk documents into `data/chunks.json`
2. `python embed.py` — embed chunks into `data/embeddings.npy`
3. `python extract.py` — extract entity/relationship triples into `data/triples.json`
4. `python entity_resolution.py` — merge duplicate entities into `data/resolved_triples.json`
5. `python build_graph.py` — build the knowledge graph into `data/graph.json`

## Usage

```powershell
python ask.py "your question here"
```

Combines vector retrieval and graph traversal, generates an answer with the local LLM, and prints sources for every fact used.
