"""
check_models.py
----------------
Prints every model your Groq API key can actually access right now.

Model availability on Groq shifts (deprecations, account-level permissions,
free-tier restrictions), so if rag_chain.py or eval/evaluate.py ever throws
a 404 "model_not_found" error, run this first instead of guessing a new
model name:

    python -m src.check_models
"""

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY not set — check your .env file.")

    client = Groq(api_key=api_key)
    models = client.models.list()

    print(f"{'MODEL ID':45s} {'ACTIVE':8s} OWNED BY")
    print("-" * 70)
    for m in sorted(models.data, key=lambda x: x.id):
        print(f"{m.id:45s} {str(m.active):8s} {m.owned_by}")

    print(f"\n{len(models.data)} models available to this API key.")
    print("Set LLM_MODEL in src/rag_chain.py (and eval/evaluate.py) to any ID above.")


if __name__ == "__main__":
    main()
