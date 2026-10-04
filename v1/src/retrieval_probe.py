import asyncio
from llama_index.core import Settings, StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers.fusion_retriever import QueryFusionRetriever, FUSION_MODES
from llama_index.core.indices.property_graph import VectorContextRetriever
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

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
graph_retriever = index.as_retriever(sub_retrievers=[vector_retriever], include_text=False)

retriever = QueryFusionRetriever(
    retrievers=[graph_retriever, vec_store_retriever, bm25_retriever],
    mode=FUSION_MODES.RECIPROCAL_RANK,
    num_queries=1,
    similarity_top_k=15,
)

reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=5)

QUERIES = [
    "Is there an alternative to a vector database for powering semantic search?",
    "What role does PostgreSQL play as a database option?",
]

async def probe(query):
    print("=" * 80)
    print("QUERY:", query)
    fused = await retriever.aretrieve(query)
    print(f"-- {len(fused)} pre-rerank fused results --")
    for r in fused:
        snippet = r.node.text.replace("\n", " ")[:160]
        print(f"{r.score:.4f}  {snippet}")
    reranked = reranker.postprocess_nodes(fused, query_str=query)
    print(f"-- top {len(reranked)} after rerank --")
    for r in reranked:
        snippet = r.node.text.replace("\n", " ")[:160]
        print(f"{r.score:.4f}  {snippet}")
    print()

async def main():
    for q in QUERIES:
        await probe(q)

asyncio.run(main())
