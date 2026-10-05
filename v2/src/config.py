CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

LLM_PROVIDER = "groq"  # "ollama" or "groq" -- the single switch between local and hosted

OLLAMA_BASE_URL = "https://unleaded-alto-sesame.ngrok-free.dev"
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/chat"
NGROK_HEADERS = {"ngrok-skip-browser-warning": "true"}

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

MODEL_NAMES = {
    "ollama": {
        "context": "qwen2.5:3b",
        "classify": "qwen2.5:3b",
        "generation": "qwen2.5:3b",
        "verify": "llama3.1:8b",
        "eval": "llama3.1:8b",
    },
    "groq": {
        "context": "openai/gpt-oss-20b",
        "classify": "openai/gpt-oss-20b",
        "generation": "openai/gpt-oss-120b",
        "verify": "openai/gpt-oss-120b",
        "eval": "openai/gpt-oss-120b",
    },
}

TEMPERATURE = 0

CONTEXT_PROMPT_TEMPLATE = """Here is the full document:
<document>
{whole_document}
</document>

Here is a specific chunk from that document:
<chunk>
{chunk}
</chunk>

Write a short, 1-2 sentence context statement that situates this chunk within the overall document, to help retrieve it correctly later. Answer with only the context statement, nothing else."""

RETRIEVAL_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHROMA_DB_PATH = "data/chroma_db"
CHROMA_COLLECTION_NAME = "deltacore"

RETRIEVAL_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


ROUTER_SYSTEM_PROMPT = """You are a query classifier for a question-answering system. Given a question, decide whether it is SIMPLE or COMPLEX.

SIMPLE: a single direct fact lookup, answerable from one place without combining multiple facts.
COMPLEX: requires combining facts from more than one place, involves an exclusion or comparison (e.g. "besides X", "other than X"), or requires multiple reasoning steps.

Respond with exactly one word: SIMPLE or COMPLEX.

Examples:
Question: What university did Michael Jordan attend?
Answer: SIMPLE

Question: Besides Michael Jordan, which player from the Chicago Bulls' second three-peat had already won a championship with a different team under a different coach?
Answer: COMPLEX

Question: How many championships did Phil Jackson win as a head coach?
Answer: SIMPLE

Question: Which NBA team originally drafted the Chicago Bulls player whose jersey number 33 is retired?
Answer: COMPLEX"""

SIMPLE_TOP_K = 10
SIMPLE_RERANK_TOP_N = 5
COMPLEX_TOP_K = 30

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"