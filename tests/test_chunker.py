import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.chunker import HierarchicalChunker


SAMPLE_TEXT = """
## Section One

This is the first paragraph of section one. It contains financial information.

This is the second paragraph with more details about earnings and revenue.

## Section Two

Trade confirmation details including CUSIP and settlement date information.
"""


def test_chunker_returns_chunks():
    chunker = HierarchicalChunker()
    chunks = chunker.chunk(SAMPLE_TEXT, doc_id="test-doc")
    assert len(chunks) > 0


def test_chunks_have_required_fields():
    chunker = HierarchicalChunker()
    chunks = chunker.chunk(SAMPLE_TEXT, doc_id="test-doc")
    for chunk in chunks:
        assert chunk.text
        assert chunk.chunk_id
        assert chunk.doc_id == "test-doc"
        assert chunk.level in ("section", "paragraph", "sentence")


def test_chunk_ids_are_unique():
    chunker = HierarchicalChunker()
    chunks = chunker.chunk(SAMPLE_TEXT, doc_id="test-doc")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_large_section_splits_to_paragraphs():
    long_section = "\n\n".join([f"Paragraph {i} " + "word " * 60 for i in range(5)])
    chunker = HierarchicalChunker(max_section_tokens=100)
    chunks = chunker.chunk(long_section, doc_id="large-doc")
    para_chunks = [c for c in chunks if c.level == "paragraph"]
    assert len(para_chunks) > 0
