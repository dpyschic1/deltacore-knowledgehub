import re

from llm_client import chat
from generate import generate_answer
from retrieve import retrieve_chunks


VERIFY_SINGLE_SYSTEM_PROMPT = """You are a fact-checking assistant. You will be given a question, a proposed answer to that question, and one source. Decide whether the source provides evidence supporting that the proposed answer is correct for the question.

Treat a partial reference to a person (such as a surname alone, e.g. "Rodman") as referring to the same person as their full name (e.g. "Dennis Rodman") unless the source gives a specific reason to think they are different people.

Respond with exactly one line: SUPPORTED or NOT SUPPORTED -- <brief reason>."""


def extract_final_answer(answer_text):
    final_line_match = re.search(r"Final answer:.*", answer_text)
    if not final_line_match:
        return None, []
    final_line = final_line_match.group()
    citation_numbers = [int(n) for n in re.findall(r"\[(\d+)\]", final_line)]
    clean_answer = re.sub(r"Final answer:\s*", "", final_line)
    clean_answer = re.sub(r"\[\d+\]", "", clean_answer)
    clean_answer = clean_answer.strip(" *_`").strip()
    return clean_answer, citation_numbers


def verify_single_citation(question, answer, source_text, doc_id):
    labeled_source = f"(From document: {doc_id})\n{source_text}"
    user_prompt = f"Question:\n{question}\n\nProposed answer:\n{answer}\n\nSource:\n{labeled_source}"
    return chat("verify", VERIFY_SINGLE_SYSTEM_PROMPT, user_prompt)


def verify_citations(question, answer_text, citation_map):
    answer, citation_numbers = extract_final_answer(answer_text)
    if answer is None:
        return "No final answer line found to verify."

    results = []
    for n in citation_numbers:
        chunk = citation_map[n]
        verdict = verify_single_citation(question, answer, chunk["text"], chunk["doc_id"])
        results.append(f"[{n}]: {verdict}")
    return "\n".join(results)


if __name__ == "__main__":
    q = "Besides Michael Jordan, which player from the Chicago Bulls' second three-peat (1996-1998) had already won an NBA championship with a different team under a different head coach?"
    retrieved_chunks = retrieve_chunks(q)
    answer_text, citation_map, source_blocks = generate_answer(q, retrieved_chunks)
    print("ANSWER:\n", answer_text)
    print("\nVERIFICATION:\n", verify_citations(q, answer_text, citation_map))
