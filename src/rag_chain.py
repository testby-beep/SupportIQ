"""
rag_chain.py
------------
Builds the retrieval-augmented generation chain:
  user question -> retrieve top-k relevant chunks from ChromaDB
                 -> stuff them into a prompt
                 -> Groq LLM generates a grounded answer with citations

This module exposes `answer_question()`, used by both the FastAPI backend
(api.py) and the Streamlit UI (app.py), so retrieval logic lives in one place.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

PERSIST_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"  # fast, cheap, and confirmed available on Groq's free tier
                                    # (llama-3.1-8b-instant was returning 404 model_not_found —
                                    # if you hit that again, run the check below to see what
                                    # your account can actually access)
TOP_K = 4

SYSTEM_PROMPT = """You are SupportIQ, a customer support assistant for CloudDesk.

Answer the user's question using ONLY the context provided below. Follow these rules:
1. If the answer is fully contained in the context, answer clearly and concisely.
2. If the context only partially covers the question, answer what you can and say
   what's missing.
3. If the context does not contain the answer at all, say "I don't have that
   information in my knowledge base" — do NOT make up an answer.
4. Keep answers concise (2-5 sentences) unless the question needs a step-by-step list.
5. Do not mention "the context" or "the documents" in your answer — answer naturally,
   as a support agent would.

Context:
{context}
"""


def _get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_name="supportiq_faq",
    )


def _get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your key "
            "(get one free at https://console.groq.com/keys)."
        )
    return ChatGroq(model=LLM_MODEL, api_key=api_key, temperature=0.2)


def _format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {os.path.basename(d.metadata.get('source', 'unknown'))}]\n{d.page_content}"
        for d in docs
    )


def answer_question(question: str) -> dict:
    """
    Runs the full RAG pipeline for a single question.

    Returns:
        {
            "answer": str,
            "sources": list[str],   # unique source filenames used
            "retrieved_chunks": list[dict],  # for debugging/eval
        }
    """
    vector_store = _get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    retrieved_docs = retriever.invoke(question)

    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    chain = (
        {
            "context": lambda x: _format_docs(retrieved_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    # langchain-core 1.x's StrOutputParser returns a `str` subclass
    # (TextAccessor) rather than a plain str. It behaves like a string
    # everywhere (isinstance, concatenation, etc.) but libraries that do a
    # strict type check — Presidio in guardrails.py being one — reject it.
    # Coerce to plain str here, once, so nothing downstream has to know
    # this quirk exists.
    answer = str(answer)

    sources = sorted({os.path.basename(d.metadata.get("source", "unknown")) for d in retrieved_docs})

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": [
            {"source": os.path.basename(d.metadata.get("source", "unknown")), "content": d.page_content}
            for d in retrieved_docs
        ],
    }


if __name__ == "__main__":
    # Quick manual smoke test from the command line:
    #   python -m src.rag_chain
    test_question = "Can I get a refund if I just subscribed?"
    result = answer_question(test_question)
    print("Q:", test_question)
    print("A:", result["answer"])
    print("Sources:", result["sources"])
