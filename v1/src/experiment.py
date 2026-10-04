import asyncio
from llama_index.core import Settings, StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers.fusion_retriever import QueryFusionRetriever, FUSION_MODES
from llama_index.core.indices.property_graph import VectorContextRetriever
from llama_index.core.schema import NodeRelationship
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from generate import generate_answer

# Not actually called by any retriever now that LLMSynonymRetriever is gone --
# but PropertyGraphIndex/as_retriever() still eagerly resolves Settings.llm at
# construction time, and its global default (if never set) is OpenAI, which
# fails immediately without an API key. Setting this just prevents that
# fallback; nothing here actually sends it a request.
Settings.llm = Ollama(model="qwen2.5:3b", request_timeout=300, temperature=0)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    query_instruction="Represent this sentence for searching relevant passages:",
)

storage_context = StorageContext.from_defaults(persist_dir="../data/llama_graph_storage")
index = load_index_from_storage(storage_context)
nodes = list(index.docstore.docs.values())

vec_store_index = VectorStoreIndex(nodes=nodes, embed_model=Settings.embed_model)
vec_store_retriever = vec_store_index.as_retriever(similarity_top_k=15)
bm25_retriever = BM25Retriever(nodes=nodes, similarity_top_k=15)

vector_retriever = VectorContextRetriever(
    graph_store=index.property_graph_store,
    vector_store=index.vector_store,
    embed_model=Settings.embed_model,
    include_text=False,
)

reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=5)

# LLMSynonymRetriever deliberately dropped here: proven unreliable across
# multiple fix attempts (base_url, event loop policy, nest_asyncio, async-native
# restructuring, fresh-client-per-call, temperature=0.3) -- root cause is
# greedy/low-temperature decoding on this model being prone to repetition-loop
# hangs on its internal synonym-generation prompt, which is inherently
# probabilistic and can't be reliably eliminated by configuration alone.
# VectorContextRetriever (graph entity similarity, no LLM calls) + vector +
# BM25 still give a real three-way hybrid without touching that failure mode.
graph_only_retriever = index.as_retriever(sub_retrievers=[vector_retriever], include_text=False)
hybrid_retriever = QueryFusionRetriever(
    retrievers=[graph_only_retriever, vec_store_retriever, bm25_retriever],
    mode=FUSION_MODES.RECIPROCAL_RANK,
    num_queries=1,
    similarity_top_k=15,
)

def node_to_chunk_dict(node_with_score, docstore):
    node = node_with_score.node
    doc_id = node.metadata.get("doc_id")
    chunk_index = node.metadata.get("chunk_index")

    if doc_id is None:
        source_info = node.relationships.get(NodeRelationship.SOURCE)
        if source_info is not None:
            source_node = docstore.get_node(source_info.node_id, raise_error=False)
            if source_node is not None:
                doc_id = source_node.metadata.get("doc_id", "graph")
                chunk_index = source_node.metadata.get("chunk_index", 0)
        if doc_id is None:
            doc_id, chunk_index = "graph", 0
    return {
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "text": node.text,
    }

async def run_with_retriever(question, retriever, label):
    fused_results = await retriever.aretrieve(question)
    results = reranker.postprocess_nodes(fused_results, query_str=question)
    retrieved = [node_to_chunk_dict(r, index.docstore) for r in results]
    answer = generate_answer(question, retrieved)
    print(f"--- {label} ---")
    print(f"Context: {len(retrieved)} chunks")
    for r in retrieved:
        print(f"  [{r['doc_id']} chunk {r['chunk_index']}] {r['text'][:80].replace(chr(10), ' ')}")
    print("Answer:", answer)
    print()
    return answer

async def compare_graph_only_vs_hybrid(question):
    await run_with_retriever(question, graph_only_retriever, "graph-only (PropertyGraphIndex)")
    await run_with_retriever(question, hybrid_retriever, "hybrid (graph + vector + BM25, RRF-fused)")

async def wide_recall_probe(question, top_k=30):
    wide_hybrid = QueryFusionRetriever(
        retrievers=[graph_only_retriever, vec_store_index.as_retriever(similarity_top_k=top_k), BM25Retriever(nodes=nodes, similarity_top_k=top_k)],
        mode=FUSION_MODES.RECIPROCAL_RANK,
        num_queries=1,
        similarity_top_k=top_k,
    )
    fused = await wide_hybrid.aretrieve(question)
    print(f"-- {len(fused)} pre-rerank hybrid results (top_k={top_k}) --")
    for i, r in enumerate(fused):
        snippet = r.node.text.replace("\n", " ")[:100]
        print(f"{i+1:2d}. {r.score:.4f}  {snippet}")

flagship_question = (
    "Besides Michael Jordan, which player from the Chicago Bulls' second "
    "three-peat (1996-1998) had already won an NBA championship with a "
    "different team under a different head coach?"
)
asyncio.run(compare_graph_only_vs_hybrid(flagship_question))
asyncio.run(wide_recall_probe(flagship_question, top_k=30))

async def wide_hybrid_answer(question, top_k=30, rerank_top_n=15):
    wide_hybrid = QueryFusionRetriever(
        retrievers=[graph_only_retriever, vec_store_index.as_retriever(similarity_top_k=top_k), BM25Retriever(nodes=nodes, similarity_top_k=top_k)],
        mode=FUSION_MODES.RECIPROCAL_RANK,
        num_queries=1,
        similarity_top_k=top_k,
    )
    wide_reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=rerank_top_n)
    fused = await wide_hybrid.aretrieve(question)
    results = wide_reranker.postprocess_nodes(fused, query_str=question)
    retrieved = [node_to_chunk_dict(r, index.docstore) for r in results]
    print(f"--- wide hybrid (top_k={top_k}, rerank_top_n={rerank_top_n}) ---")
    print(f"Context: {len(retrieved)} chunks")
    for r in retrieved:
        print(f"  [{r['doc_id']} chunk {r['chunk_index']}] {r['text'][:80].replace(chr(10), ' ')}")
    answer = generate_answer(question, retrieved)
    print("Bare answer:", answer)

    reasoning_question = question + " Explain your reasoning step by step, citing the specific facts that connect the answer, before giving your final answer."
    reasoning_answer = generate_answer(reasoning_question, retrieved)
    print("Reasoning trace:")
    print(reasoning_answer)

asyncio.run(wide_hybrid_answer(flagship_question, top_k=30, rerank_top_n=15))

async def vector_bm25_only_answer(question, top_k=30, rerank_top_n=15):
    no_graph_retriever = QueryFusionRetriever(
        retrievers=[vec_store_index.as_retriever(similarity_top_k=top_k), BM25Retriever(nodes=nodes, similarity_top_k=top_k)],
        mode=FUSION_MODES.RECIPROCAL_RANK,
        num_queries=1,
        similarity_top_k=top_k,
    )
    wide_reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=rerank_top_n)
    fused = await no_graph_retriever.aretrieve(question)
    results = wide_reranker.postprocess_nodes(fused, query_str=question)
    retrieved = [node_to_chunk_dict(r, index.docstore) for r in results]
    print(f"--- vector + BM25 ONLY, no graph (top_k={top_k}, rerank_top_n={rerank_top_n}) ---")
    print(f"Context: {len(retrieved)} chunks")
    for r in retrieved:
        print(f"  [{r['doc_id']} chunk {r['chunk_index']}] {r['text'][:80].replace(chr(10), ' ')}")

    reasoning_question = question + " Explain your reasoning step by step, citing the specific facts that connect the answer, before giving your final answer."
    reasoning_answer = generate_answer(reasoning_question, retrieved)
    print("Reasoning trace:")
    print(reasoning_answer)

asyncio.run(vector_bm25_only_answer(flagship_question, top_k=30, rerank_top_n=15))
