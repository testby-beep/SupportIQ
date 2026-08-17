"""
evaluate.py
-----------
Scores the RAG pipeline against a golden Q&A dataset using Ragas metrics:

- faithfulness       : does the answer avoid claims unsupported by retrieved context?
- answer_relevancy    : does the answer actually address the question asked?
- context_precision   : are the retrieved chunks relevant to the question?
- context_recall      : did retrieval surface what was needed to answer correctly?

Run:
    python -m eval.evaluate

Outputs a printed summary table and eval/eval_results.csv for tracking over time.
"""

import json
import sys
import types
from pathlib import Path

# --- Compatibility shim ------------------------------------------------
# Ragas (as of 0.4.x) unconditionally imports `ChatVertexAI` from
# `langchain_community.chat_models.vertexai` at module load time, even
# though we never use Vertex AI here. That submodule was removed from
# recent langchain-community releases (Google's integration moved to the
# standalone `langchain-google-vertexai` package), so the bare `import
# ragas` crashes with ModuleNotFoundError before we ever get a chance to
# pick which LLM we actually want (Groq).
#
# This registers empty stand-in modules so Ragas's import succeeds. It's
# safe: we never construct ChatVertexAI/VertexAI, we only use
# LangchainLLMWrapper(ChatGroq(...)). Remove this once Ragas fixes the
# upstream import (tracked as a known issue as of ragas==0.4.3).
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai_chat_stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _vertexai_chat_stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_chat_stub

from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.rag_chain import answer_question

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.csv"


def build_eval_dataset():
    """Run the RAG pipeline on every golden question and assemble a Ragas-ready dataset."""
    with open(DATASET_PATH) as f:
        golden = json.load(f)

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in golden:
        q = item["question"]
        print(f"Running: {q}")
        result = answer_question(q)

        questions.append(q)
        answers.append(result["answer"])
        contexts.append([c["content"] for c in result["retrieved_chunks"]])
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def main():
    dataset = build_eval_dataset()

    # Ragas needs an LLM (to judge faithfulness/relevancy) and embeddings
    # (for semantic similarity metrics). Reuse the same Groq model + local
    # embeddings the app already runs on, wrapped for Ragas via LangChain.
    judge_llm = LangchainLLMWrapper(ChatGroq(model="openai/gpt-oss-20b", temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    df = results.to_pandas()
    df.to_csv(RESULTS_PATH, index=False)

    print("\n" + "=" * 60)
    print("EVAL SUMMARY (mean across all questions)")
    print("=" * 60)
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric in df.columns:
            print(f"  {metric:20s}: {df[metric].mean():.3f}")
    print("=" * 60)
    print(f"\nPer-question results saved to {RESULTS_PATH}")

    # Flag any question that scored poorly on faithfulness (i.e. likely hallucination)
    if "faithfulness" in df.columns:
        weak = df[df["faithfulness"] < 0.7]
        if len(weak) > 0:
            print(f"\n⚠️  {len(weak)} question(s) scored below 0.7 on faithfulness — review these:")
            for _, row in weak.iterrows():
                print(f"  - {row['question']}")


if __name__ == "__main__":
    main()
