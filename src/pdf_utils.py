from typing import List

from pypdf import PdfReader


def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(texts)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += max(chunk_size - overlap, 1)
    return chunks


def extract_pdf_chunks(path: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    text = load_pdf_text(path)
    if not text.strip():
        return []
    return chunk_text(text, chunk_size=chunk_size, overlap=overlap)

