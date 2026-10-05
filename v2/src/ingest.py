import json
import re
import time
from pathlib import Path
from config import CHUNK_SIZE, CHUNK_OVERLAP, CONTEXT_PROMPT_TEMPLATE, LLM_PROVIDER
from llm_client import chat

# For pacing requests when sending calls to external providers
SECONDS_BETWEEN_CONTEXT_CALLS = 3.0 if LLM_PROVIDER == "groq" else 0

HEADING_RE = re.compile(r'^#{1,6}\s+.*$', re.MULTILINE)


def split_into_sections(text, is_markdown):
    if is_markdown:
        headings = list(HEADING_RE.finditer(text))
        if not headings:
            return split_into_paragraphs(text)
        sections = []
        for i , m in enumerate(headings): #iterate over headings and extract bounded paragraphs 
            start = m.start()
            end = headings[i + 1].start() if i+1 < len(headings) else len((text))
            if text[m.end():end].strip():
                sections.append({'heading': m.group().strip(), 'start': start, 'end': end, 'text': text[m.end():end]})
        return sections
    else:
        return split_into_paragraphs(text)

def split_into_paragraphs(text):
    sections = []
    start = 0
    for m in re.finditer(r'\n\s*\n', text): #this is useful for both text and md files, we want to extract individual paragraphs if not part of a heading
        end = m.start()
        if text[start:end].strip():
            sections.append({'heading': None, 'start': start, 'end': end, 'text': text[start:end]})
        start = m.end()
    if text[start:].strip():
        sections.append({'heading': None, 'start': start, 'end': len(text), 'text': text[start:]})
    return sections

def chunk_document(text, is_markdown, chunk_size, overlap):
    sections = split_into_sections(text, is_markdown)
    chunks = []
    for section in sections:
        if len(section['text']) <= chunk_size:
            chunks.append({'start_char': section['start'], 'end_char': section['end'],
                            'heading': section['heading'], 'chunk': section['text']})
        else:
            for sc in chunk_text(section['text'], chunk_size, overlap):
                chunks.append({'start_char': section['start'] + sc['start_char'],
                                'end_char': section['start'] + sc['end_char'],
                                'heading': section['heading'], 'chunk': sc['chunk']})
    return chunks

#for chunking based on a fixed token size and overlapping boundaries so no text gets missed
def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            boundary = end
            while boundary > start and not text[boundary].isspace():
                boundary -= 1
            if boundary > start:
                end = boundary
        chunks.append({'start_char': start, 'end_char': end, 'chunk': text[start:end]})
        if(end >= n):
            break
        next_start = end - overlap
        while next_start < end and not text[next_start].isspace():
            next_start += 1
        start = next_start + 1 if next_start < end else end
    return chunks

#context is helpful downstream to better place the chunk in the document, specially useful for multi-hop reasoning
def generate_context(whole_document_text, chunk_text):
    return chat(
        "context",
        "You are a helpful assistant that writes concise context summaries.",
        CONTEXT_PROMPT_TEMPLATE.format(whole_document=whole_document_text, chunk=chunk_text),
    )


def main():
    targetPath = Path(__file__).resolve().parent.parent / "data" / "chunks.json"

    existing_by_key = {}
    if targetPath.exists():
        with open(targetPath, "r", encoding="utf-8") as f:
            for c in json.load(f):
                existing_by_key[(c['doc_id'], c['chunk_index'], c['text'])] = c
        print(f"Found {len(existing_by_key)} existing chunks -- will reuse any that are unchanged.")

    all_chunks = []

    def save_progress():
        with open(targetPath, "w", encoding="utf-8") as target:
            json.dump(all_chunks, target, indent=4, ensure_ascii=False)

    path = Path(__file__).resolve().parent.parent / "docs"
    for file in (list(path.glob('**/*.md')) + list(path.glob('**/*.txt'))):
        with file.open(encoding="utf-8") as f:
            text = f.read()
            chunked = chunk_document(text, file.suffix == '.md', CHUNK_SIZE, CHUNK_OVERLAP)
            for i,p in enumerate(chunked):
                doc_id = str(file.relative_to(path))
                chunk_text_value = f"{p['heading']}\n\n{p['chunk']}" if p['heading'] else p['chunk']
                key = (doc_id, i, chunk_text_value)

                if key in existing_by_key:
                    print(f"Reusing existing context: {file.name} chunk {i}")
                    all_chunks.append(existing_by_key[key])
                else:
                    print(f"Generating context: {file.name} chunk {i}")
                    generated_blurb = generate_context(text, p['chunk'])
                    if SECONDS_BETWEEN_CONTEXT_CALLS:
                        time.sleep(SECONDS_BETWEEN_CONTEXT_CALLS)
                    all_chunks.append({
                        'doc_id': doc_id,
                        'source_path': str(file.absolute()),
                        'chunk_index': i,
                        'start_char': p['start_char'],
                        'end_char': p['end_char'],
                        'text': chunk_text_value,
                        'context': generated_blurb
                    })

                save_progress()

    save_progress()

main()