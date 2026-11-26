import pytest
from rag.ingestion import chunk_text

def test_chunking_metadata():
    chunks = chunk_text("A " * 1000, "sample.txt", chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert chunks[0]["metadata"]["source"] == "sample.txt"

def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello", "x.txt", chunk_size=100, overlap=100)
