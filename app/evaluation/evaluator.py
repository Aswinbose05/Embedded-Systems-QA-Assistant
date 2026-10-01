import os
from typing import Dict, List, Optional
import pandas as pd
from openai import OpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from app.core.config import settings
from app.core.logging import get_logger
from app.rag.retriever import rag_retriever
from app.gateway.llm_gateway import llm_gateway
from app.rag.pipeline import format_rag_user_prompt, RAG_SYSTEM_PROMPT

logger = get_logger("evaluation")


class RAGEvaluator:
    """
    Independent Ragas evaluation module.
    Runs separately from the production API to avoid overhead on live requests.
    Computes Faithfulness, Answer Relevancy, Context Precision, and Context Recall.
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        eval_model: Optional[str] = None,
    ):
        self.groq_api_key = groq_api_key or settings.GROQ_API_KEY
        self.eval_model = eval_model or settings.FALLBACK_MODEL or "openai/gpt-oss-20b"

    def _setup_evaluator_llm(self):
        groq_client = OpenAI(
            api_key=self.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        return llm_factory(self.eval_model, client=groq_client)

    def _setup_evaluator_embeddings(self):
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            client = OpenAI(api_key=openai_key)
            return embedding_factory("openai", model="text-embedding-3-small", client=client, interface="modern")
        # Fallback to local HuggingFace embeddings
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas.embeddings.base import LangchainEmbeddingsWrapper
        hf_emb = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        return LangchainEmbeddingsWrapper(hf_emb)

    def run_evaluation(
        self,
        dataset_path: str = "evaluation/test_dataset.csv",
        result_path: str = "evaluation/ragas_results.csv",
        summary_path: str = "evaluation/ragas_summary.csv",
        top_k: int = 3,
    ) -> Dict[str, float]:
        """
        Executes end-to-end RAG generation for dataset questions and scores them with Ragas.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

        df = pd.read_csv(dataset_path)
        required_cols = ["question", "reference_answer"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Dataset missing required column '{col}'. Required: {required_cols}")

        logger.info("Loaded %d evaluation samples from %s", len(df), dataset_path)

        samples: List[Dict] = []
        for idx, row in df.iterrows():
            question = str(row["question"])
            reference = str(row["reference_answer"])

            # 1. Retrieve
            raw_docs, _ = rag_retriever.retrieve(question, top_k=top_k)
            retrieved_contexts = [doc.page_content for doc in raw_docs]

            # 2. Prompt & Generate through Gateway
            context_str = rag_retriever.build_context(raw_docs)
            prompt = format_rag_user_prompt(context_str, question)
            messages = [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            response = llm_gateway.generate(messages=messages, temperature=0.0)

            samples.append({
                "user_input": question,
                "retrieved_contexts": retrieved_contexts if retrieved_contexts else [""],
                "response": response.content,
                "reference": reference,
            })
            logger.info("Processed sample %d/%d: %s", idx + 1, len(df), question[:40])

        eval_dataset = EvaluationDataset.from_list(samples)
        evaluator_llm = self._setup_evaluator_llm()
        evaluator_embeddings = self._setup_evaluator_embeddings()

        metrics = [
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            ContextPrecision(llm=evaluator_llm),
            ContextRecall(llm=evaluator_llm),
        ]

        logger.info("Running Ragas evaluation metrics...")
        result = evaluate(dataset=eval_dataset, metrics=metrics)
        result_df = result.to_pandas()

        # Save results
        os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
        result_df.to_csv(result_path, index=False)
        logger.info("Saved detailed evaluation results to %s", result_path)

        # Average summary
        metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        summary = {}
        for col in metric_cols:
            matching = [c for c in result_df.columns if col in c.lower()]
            if matching:
                summary[col] = float(result_df[matching[0]].mean())

        summary_df = pd.DataFrame([{"metric": k, "average_score": v} for k, v in summary.items()])
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved summary evaluation metrics to %s", summary_path)

        return summary
