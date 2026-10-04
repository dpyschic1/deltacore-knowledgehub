from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor, PropertyGraphIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path
import json
Settings.llm = Ollama(model="qwen2.5:3b", request_timeout=300)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", query_instruction="Represent this sentence for searching relevant passages:")

def main():
    data_path = Path(__file__).resolve().parent.parent.parent / "data"
    with open(data_path / "chunks.json", "r", encoding="utf-8") as c:
        chunks = json.load(c)

    textnodes = []
    for chunk in chunks:
        node = TextNode(
            text=chunk["text"], 
            id_=f"{chunk["doc_id"]}-{chunk["chunk_index"]}",
            metadata={"doc_id":chunk["doc_id"], "chunk_index":chunk["chunk_index"]})
        textnodes.append(node)

    extractor = SimpleLLMPathExtractor(llm=Settings.llm, num_workers=1)
    index = PropertyGraphIndex(nodes=textnodes, kg_extractors=[extractor], show_progress=True)
    index.storage_context.persist(data_path / "llama_graph_storage")

main()