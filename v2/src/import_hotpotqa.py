import json
import re
from pathlib import Path
from datasets import load_dataset

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
CANDIDATE_QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "hotpotqa_candidate_questions.json"

# Rough starting point for ~500 chunks after ingest.py's chunker runs. Each
# question drags in ~10 context paragraphs (2 gold + ~8 distractor), with
# light overlap across questions, and most paragraphs are short enough to
# become a single chunk -- but this is an estimate, not exact. After running
# this + ingest.py, check data/chunks.json's length and adjust NUM_QUESTIONS
# up or down, then rerun both.
NUM_QUESTIONS = 50

# How many (question, answer) pairs to export for manual review as candidate
# additions to deepeval_eval.py's TEST_QUESTIONS. Spot-check each one against
# its supporting_titles' docs before trusting it -- same rigor as the
# hand-verified Chicago Bulls questions, even though HotpotQA's annotations
# are independently constructed, not something we authored ourselves.
NUM_CANDIDATE_TEST_QUESTIONS = 10


def sanitize_filename(title):
    name = title.strip().lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s-]+", "_", name)
    return name or "untitled"


def main():
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    selected = dataset.select(range(min(NUM_QUESTIONS, len(dataset))))
    print(f"Using {len(selected)} of {len(dataset)} total questions in the dataset.")

    unique_titles = {}
    for item in selected:
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            if title not in unique_titles:
                unique_titles[title] = sentences

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for title, sentences in unique_titles.items():
        # HotpotQA's sentence splits already carry their own spacing, so a
        # plain join reconstructs the original paragraph -- no extra spaces
        # need to be inserted between them.
        paragraph = "".join(sentences).strip()
        if not paragraph:
            continue
        filename = f"hotpotqa_{sanitize_filename(title)}.md"
        doc_text = f"# {title}\n\n{paragraph}\n"
        with open(DOCS_DIR / filename, "w", encoding="utf-8") as out:
            out.write(doc_text)
        written += 1

    print(f"Wrote {written} new documents to {DOCS_DIR}")

    candidates = []
    for item in selected.select(range(min(NUM_CANDIDATE_TEST_QUESTIONS, len(selected)))):
        candidates.append({
            "question": item["question"],
            "answer": item["answer"],
            "supporting_titles": sorted(set(item["supporting_facts"]["title"])),
            "type": item.get("type"),
            "level": item.get("level"),
        })

    with open(CANDIDATE_QUESTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(candidates)} candidate test questions to {CANDIDATE_QUESTIONS_PATH}")
    print("Review these by hand -- spot-check each answer against its supporting_titles' docs -- "
          "before adding any to deepeval_eval.py's TEST_QUESTIONS.")


if __name__ == "__main__":
    main()
