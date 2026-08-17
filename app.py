"""
app.py
------
Streamlit chat interface for SupportIQ, the CloudDesk RAG support assistant.

Run:
    streamlit run app.py
"""

import streamlit as st

from src.guardrails import check_input, check_output
from src.observability import traced_answer_question

st.set_page_config(page_title="SupportIQ", page_icon="💬", layout="centered")

st.title("💬 SupportIQ")
st.caption("RAG-powered support assistant for CloudDesk — answers are grounded in the FAQ knowledge base, with sources cited.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything about billing, your account, or CloudDesk integrations.", "sources": []}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("📄 Sources: " + ", ".join(msg["sources"]))

if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            input_check = check_input(question)
            if not input_check.allowed:
                st.error(input_check.reason)
            else:
                try:
                    result = traced_answer_question(input_check.cleaned_text)
                    output_check = check_output(result["answer"])
                    final_answer = output_check.cleaned_text if output_check.allowed else result["answer"]

                    st.markdown(final_answer)
                    if result["sources"]:
                        st.caption("📄 Sources: " + ", ".join(result["sources"]))
                    if input_check.pii_found or output_check.pii_found:
                        st.caption("🔒 PII was detected and redacted in this exchange.")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": final_answer, "sources": result["sources"]}
                    )
                except RuntimeError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

with st.sidebar:
    st.header("About this project")
    st.markdown(
        """
        **SupportIQ** is a retrieval-augmented generation (RAG) assistant that answers
        customer support questions grounded in a real knowledge base — not the LLM's
        general training data.

        **Pipeline:**
        1. FAQ docs are chunked with `RecursiveCharacterTextSplitter`
        2. Chunks are embedded locally via `sentence-transformers/all-MiniLM-L6-v2`
        3. Embeddings are stored in **ChromaDB**
        4. On each question, top-k relevant chunks are retrieved
        5. **Groq (Llama 3.1)** generates a grounded answer, citing sources

        Built with LangChain, ChromaDB, FastAPI, and Streamlit.
        """
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
