"""
tests/test_ingest.py
---------------------
Tests the chunking logic in isolation. Doesn't touch embeddings or the
vector store, so it's fast and has no external dependencies.
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader

from src.ingest import DATA_DIR, chunk_documents


def test_documents_load():
    loader = DirectoryLoader(str(DATA_DIR), glob="*.txt", loader_cls=TextLoader, show_progress=False)
    docs = loader.load()
    assert len(docs) >= 3  # billing, account_security, technical_integrations


def test_chunking_produces_nonempty_chunks():
    loader = DirectoryLoader(str(DATA_DIR), glob="*.txt", loader_cls=TextLoader, show_progress=False)
    docs = loader.load()
    chunks = chunk_documents(docs)

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.page_content.strip()) > 0
        assert "source" in chunk.metadata


def test_chunks_are_within_size_bounds():
    loader = DirectoryLoader(str(DATA_DIR), glob="*.txt", loader_cls=TextLoader, show_progress=False)
    docs = loader.load()
    chunks = chunk_documents(docs)

    # Allow some slack over CHUNK_SIZE since splitter can't always cut exactly
    # at the boundary without breaking words/separators.
    for chunk in chunks:
        assert len(chunk.page_content) <= 700
