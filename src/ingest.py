"""
ingest.py
---------
Loads raw support docs, splits them into semantically coherent chunks,
embeds them with a local Sentence-Transformers model, and persists them
into a local ChromaDB vector store.

Run:
    python -m src.ingest
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "faq_docs"
PERSIST_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking params: FAQ docs are Q/A pairs, so mid-size chunks with overlap
# keep each answer intact rather than cutting a sentence in half.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75


def load_documents():
    """Load every .txt file in data/faq_docs, tagging each with its source filename."""
    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} source documents from {DATA_DIR}")
    return docs


def chunk_documents(docs):
    """Split documents into overlapping chunks using recursive character splitting."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\nQ:", "\n\n", "\n", " ", ""],  # prefer splitting between Q&A pairs
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_vector_store(chunks):
    """Embed chunks locally (free, no API cost) and persist to ChromaDB on disk."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
        collection_name="supportiq_faq",
    )
    print(f"Persisted {len(chunks)} embedded chunks to {PERSIST_DIR}")
    return vector_store


def main():
    docs = load_documents()
    if not docs:
        raise SystemExit(f"No .txt files found in {DATA_DIR}. Add some FAQ docs first.")
    chunks = chunk_documents(docs)
    build_vector_store(chunks)
    print("\nIngestion complete. Vector store ready for querying.")


if __name__ == "__main__":
    main()
