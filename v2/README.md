# PythonBasedRag v2

A from-scratch RAG pipeline built with standard, industry-recommended components rather than hand-rolled equivalents: Chroma for vector storage, BM25 for sparse retrieval, Reciprocal Rank Fusion for hybrid search, adaptive query routing, contextual chunk enrichment, grounded generation with inline citations, LLM-based citation verification, and DeepEval for automated evaluation. Pluggable between a local Ollama model and Groq's hosted API via a single config switch.

This is a redesign of the original hand-built pipeline in `../v1`, which remains untouched as reference.

## Prerequisites

- Python 3.11+ (developed against 3.14)
- Either:
  - **Local**: [Ollama](https://ollama.com) installed and running, with `qwen2.5:3b` and `llama3.1:8b` pulled, or
  - **Hosted**: a free [Groq](https://console.groq.com) API key

## Setup

```powershell
python -m venv ragExpirement
ragExpirement\Scripts\activate
pip install -r requirements.txt
```

If using Groq, set your key before running anything:
```powershell
$env:GROQ_API_KEY = "your-key-here"
```

## Provider switching

`src/config.py` has one switch, `LLM_PROVIDER = "ollama"` or `"groq"`. Every script resolves its model through `llm_client.chat(role, ...)`, which looks up the right model per role (`context`, `classify`, `generation`, `verify`, `eval`) for whichever provider is active — nothing else needs to change when switching.

## Project layout

```
docs/   source documents (.md) to index
src/    pipeline scripts
data/   generated artifacts (chunks, Chroma vector DB) — gitignored entirely,
        reproducible by rerunning the pipeline scripts against docs/
```

`docs/hotpotqa_*.md` is also gitignored — those are mechanically regenerable via `import_hotpotqa.py` pulling from a stable public dataset, not hand-authored content. The original hand-verified docs (`michael_jordan.md` etc.) stay tracked.

## Pipeline

Run in order from `src/` whenever `docs/` changes:

0. *(optional)* `python import_hotpotqa.py` — pull a batch of Wikipedia paragraphs + their multi-hop question/answer annotations from the [HotpotQA](https://hotpotqa.github.io/) dataset (via Hugging Face, no manual download needed) into `docs/hotpotqa_*.md`, for scaling up the corpus beyond the hand-authored docs. Tune `NUM_QUESTIONS` at the top of the script to control how much it pulls in.
1. `python ingest.py` — chunk documents, generate a situating context blurb per chunk (via LLM), write `data/chunks.json`
2. `python embed.py` — embed `context + text` per chunk into a persistent Chroma collection at `data/chroma_db/`

## Asking a question

```powershell
python ask.py "your question here"
```

Runs the question through adaptive routing + hybrid retrieval, generates a grounded answer with inline `[n]` citations, and verifies the final answer's citations against their sources.

## Evaluation

```powershell
python deepeval_eval.py
```

Runs a small hand-verified question set through the full pipeline and scores it with Faithfulness, Answer Relevancy, Contextual Precision, and Contextual Recall. If using Groq, these env vars help avoid free-tier rate-limit failures on top of the pacing already built into the script:
```powershell
$env:DEEPEVAL_RETRY_MAX_ATTEMPTS = "5"
$env:DEEPEVAL_RETRY_CAP_SECONDS = "65"
```

