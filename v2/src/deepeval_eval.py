from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.evaluate.configs import AsyncConfig, CacheConfig
from retrieve import retrieve_chunks
from generate import generate_answer
from config import LLM_PROVIDER, MODEL_NAMES, OLLAMA_BASE_URL, GROQ_BASE_URL
from llm_client import GROQ_API_KEY
import time

eval_model_name = MODEL_NAMES[LLM_PROVIDER]["eval"]

if LLM_PROVIDER == "ollama":
    from deepeval.models import OllamaModel
    judge_model = OllamaModel(model=eval_model_name, base_url=OLLAMA_BASE_URL, temperature=0)
elif LLM_PROVIDER == "groq":
    from deepeval.models import OpenAIModel
    judge_model = OpenAIModel(
        model=eval_model_name,
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        temperature=0,
        generation_kwargs={"max_tokens": 4096},
    )
    judge_model.model_data.supports_json = True
else:
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}")

TEST_QUESTIONS = [
    ("What university did Michael Jordan attend?", "University of North Carolina at Chapel Hill"),
    ("How many total NBA championships did Phil Jackson win as a head coach?", "Eleven"),
    ("Which NBA team originally drafted the Chicago Bulls player whose jersey number 33 is retired?", "Seattle SuperSonics"),
    ("Besides Michael Jordan, which player from the Chicago Bulls' second three-peat (1996-1998) had already won an NBA championship with a different team under a different head coach?", "Dennis Rodman"),
    ("Who was the head coach of the Chicago Bulls during their 72-win 1995-96 season?", "Phil Jackson"),
]

SECONDS_BETWEEN_QUESTIONS = 30
EVAL_MAX_CONTEXT_CHUNKS = 10

metrics = [
    FaithfulnessMetric(model=judge_model),
    AnswerRelevancyMetric(model=judge_model),
    ContextualPrecisionMetric(model=judge_model),
    ContextualRecallMetric(model=judge_model),
]

all_results = []
for i, (question, ground_truth) in enumerate(TEST_QUESTIONS):
    print(f"[{i + 1}/{len(TEST_QUESTIONS)}] retrieving chunks for: {question}")
    retrieved_chunks = retrieve_chunks(question)
    print("generating answer ...")
    answer_text, _, _ = generate_answer(question, retrieved_chunks)

    test_case = LLMTestCase(
        input=question,
        actual_output=answer_text,
        expected_output=ground_truth,
        retrieval_context=[c["text"] for c in retrieved_chunks[:EVAL_MAX_CONTEXT_CHUNKS]],
    )

    print(f"evaluating question {i + 1}/{len(TEST_QUESTIONS)} ...")
    result = evaluate(
        test_cases=[test_case],
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False),
    )
    all_results.append(result)

    if i < len(TEST_QUESTIONS) - 1:
        print(f"sleeping {SECONDS_BETWEEN_QUESTIONS}s before next question to stay under rate limits ...")
        time.sleep(SECONDS_BETWEEN_QUESTIONS)
