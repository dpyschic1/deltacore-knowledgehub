import json
import re
from pathlib import Path
from datasets import load_dataset

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
CANDIDATE_QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "hotpotqa_candidate_questions.json"

NUM_QUESTIONS = 40

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

    removed = 0
    for old_file in DOCS_DIR.glob("hotpotqa_*.md"):
        old_file.unlink()
        removed += 1
    if removed:
        print(f"Removed {removed} hotpotqa_* docs from a previous run.")

    written = 0
    for title, sentences in unique_titles.items():

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
