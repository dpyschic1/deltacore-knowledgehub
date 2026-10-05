import os
import time
import requests

from config import (
    LLM_PROVIDER,
    OLLAMA_URL,
    NGROK_HEADERS,
    GROQ_BASE_URL,
    MODEL_NAMES,
    TEMPERATURE,
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


def chat(role, system_prompt, user_prompt, temperature=None):
    if temperature is None:
        temperature = TEMPERATURE
    model = MODEL_NAMES[LLM_PROVIDER][role]

    if LLM_PROVIDER == "ollama":
        return _chat_ollama(model, system_prompt, user_prompt, temperature)
    if LLM_PROVIDER == "groq":
        return _chat_groq(model, system_prompt, user_prompt, temperature)
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}")


def _chat_ollama(model, system_prompt, user_prompt, temperature):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    response = requests.post(OLLAMA_URL, json=payload, headers=NGROK_HEADERS, timeout=300)
    response.raise_for_status()
    return response.json()["message"]["content"]


GROQ_MAX_RETRIES = 5
GROQ_MAX_WAIT_SECONDS = 60


class GroqSustainedRateLimitError(RuntimeError):
    pass


def _chat_groq(model, system_prompt, user_prompt, temperature):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable is not set")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    for attempt in range(GROQ_MAX_RETRIES):
        response = requests.post(f"{GROQ_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=60)
        if response.status_code == 429:
            rate_limit_headers = {
                k: v for k, v in response.headers.items() if "ratelimit" in k.lower() or k == "Retry-After"
            }
            print(f"Groq 429 headers: {rate_limit_headers}")
            print(f"Groq 429 body: {response.text[:500]}")
            retry_after = response.headers.get("Retry-After")
            wait_seconds = float(retry_after) if retry_after else (2 ** attempt)
            if wait_seconds > GROQ_MAX_WAIT_SECONDS:
                raise GroqSustainedRateLimitError(
                    f"Groq requires a {wait_seconds:.0f}s wait, past the {GROQ_MAX_WAIT_SECONDS}s cap -- "
                    "this looks like a sustained/daily quota, not a brief burst. Check usage at "
                    "console.groq.com rather than retrying; this run's progress so far is saved."
                )
            print(f"Groq rate limit hit, waiting {wait_seconds:.1f}s (attempt {attempt + 1}/{GROQ_MAX_RETRIES})")
            time.sleep(wait_seconds)
            continue
        if response.status_code >= 500:
            wait_seconds = 2 ** attempt
            if wait_seconds > GROQ_MAX_WAIT_SECONDS:
                raise RuntimeError(f"Groq server errors persisted past the {GROQ_MAX_WAIT_SECONDS}s cap.")
            print(f"Groq server error {response.status_code}, waiting {wait_seconds:.1f}s (attempt {attempt + 1}/{GROQ_MAX_RETRIES})")
            time.sleep(wait_seconds)
            continue
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    raise RuntimeError(f"Groq errors persisted after {GROQ_MAX_RETRIES} retries")
