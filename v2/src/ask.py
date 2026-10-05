import sys

from retrieve import retrieve_chunks
from generate import generate_answer
from verify import verify_citations


def main():
    if len(sys.argv) < 2:
        print('Usage: python ask.py "your question here"')
        sys.exit(1)
    question = sys.argv[1]

    retrieved_chunks = retrieve_chunks(question)
    answer_text, citation_map, _ = generate_answer(question, retrieved_chunks)

    print("Answer:")
    print(answer_text)
    print()
    print("Citation verification:")
    print(verify_citations(question, answer_text, citation_map))


if __name__ == "__main__":
    main()
