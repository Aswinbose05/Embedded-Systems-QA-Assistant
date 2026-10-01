"""
Standalone CLI script to execute Ragas evaluation on the RAG pipeline.
Usage:
    python evaluation/evaluate_rag.py
"""
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from app.evaluation.evaluator import RAGEvaluator
from app.core.config import settings

def main():
    print("=" * 70)
    print("      EMBEDDED SYSTEMS QA - RAGAS EVALUATION PIPELINE")
    print("=" * 70)
    print(f"Primary Model:    {settings.PRIMARY_MODEL}")
    print(f"Fallback Model:   {settings.FALLBACK_MODEL}")
    print(f"Embedding Model:  {settings.EMBEDDING_MODEL}")
    print(f"Chroma Directory: {settings.CHROMA_DIRECTORY}")
    print("=" * 70)

    dataset_path = "evaluation/test_dataset.csv"
    result_path = "evaluation/ragas_results.csv"
    summary_path = "evaluation/ragas_summary.csv"

    evaluator = RAGEvaluator()
    try:
        summary = evaluator.run_evaluation(
            dataset_path=dataset_path,
            result_path=result_path,
            summary_path=summary_path,
            top_k=3,
        )

        print("\n" + "=" * 70)
        print("                 EVALUATION SUMMARY")
        print("=" * 70)
        for metric, score in summary.items():
            print(f"  {metric:<30}: {score:.4f}")
        print("=" * 70)
        print(f"Detailed results: {result_path}")
        print(f"Summary metrics:  {summary_path}")
        print("=" * 70)

    except Exception as exc:
        print(f"\n❌ Evaluation failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()