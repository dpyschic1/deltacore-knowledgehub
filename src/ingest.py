import json
import re
from pathlib import Path
from config import CHUNK_SIZE, CHUNK_OVERLAP

HEADING_RE = re.compile(r'^#{1,6}\s+.*$', re.MULTILINE)


def split_into_sections(text, is_markdown):
    if is_markdown:
        headings = list(HEADING_RE.finditer(text))
        if not headings:
            return split_into_paragraphs(text)
        sections = []
        for i , m in enumerate(headings):
            start = m.start()
            end = headings[i + 1].start() if i+1 < len(headings) else len((text))
            sections.append({'heading': m.group().strip(), 'start': start, 'end': end, 'text': text[m.end():end]})
        return sections
    else:
        return split_into_paragraphs(text)

def split_into_paragraphs(text):
    sections = []
    start = 0
    for m in re.finditer(r'\n\s*\n', text):
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


def main():
    all_chunks = []
    path = Path(__file__).resolve().parent.parent / "docs"
    for file in (list(path.glob('**/*.md')) + list(path.glob('**/*.txt'))):
        with file.open(encoding="utf-8") as f:
            text = f.read()
            chunked = chunk_document(text, file.suffix == '.md', CHUNK_SIZE, CHUNK_OVERLAP)
            for i,p in enumerate(chunked):
                all_chunks.append({
                    'doc_id': str(file.relative_to(path)),
                    'source_path': str(file.absolute()),
                    'chunk_index': i,
                    'start_char': p['start_char'],
                    'end_char': p['end_char'],
                    'text': f"{p['heading']}\n\n{p['chunk']}" if p['heading'] else p['chunk']
                })
    targetPath = Path(__file__).resolve().parent.parent / "data" / "chunks.json"
    with open(targetPath, "w", encoding="utf-8") as target:
        json.dump(all_chunks, target, indent=4,ensure_ascii=False)

main()
