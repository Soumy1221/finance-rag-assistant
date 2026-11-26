from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED = {".pdf", ".txt", ".md", ".csv"}

def read_file(path):
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    if path.suffix.lower() == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8-sig", errors="replace")

def chunk_text(text, source, chunk_size=900, overlap=150):
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return [
        {"text": chunk, "metadata": {"source": source, "chunk": i}}
        for i, chunk in enumerate(splitter.split_text(text)) if chunk.strip()
    ]

def ingest_path(path):
    p = Path(path)
    return chunk_text(read_file(p), p.name)
