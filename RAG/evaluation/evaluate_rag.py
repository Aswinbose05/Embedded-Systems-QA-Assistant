import os
import pandas as pd

from dotenv import load_dotenv

# LangChain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# OpenAI-compatible client
from openai import OpenAI

# Ragas
from ragas import EvaluationDataset, evaluate
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

# ============================================================
# 1. LOAD ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in .env file"
    )

print("✅ GROQ_API_KEY loaded")


# ============================================================
# 2. CONFIGURATION
# ============================================================

CHROMA_DIRECTORY = "./chroma_db"

COLLECTION_NAME = "academic_qa"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

GROQ_MODEL = "llama-3.1-8b-instant"

TOP_K = 3

DATASET_PATH = "evaluation/test_dataset.csv"

RESULT_PATH = "evaluation/ragas_results.csv"

SUMMARY_PATH = "evaluation/ragas_summary.csv"


# ============================================================
# 3. LOAD EMBEDDINGS
# ============================================================

print("\nLoading HuggingFace embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

print("✅ Embedding model loaded")


# ============================================================
# 4. LOAD CHROMA
# ============================================================

print("\nLoading ChromaDB...")

vectordb = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIRECTORY,
    embedding_function=embeddings,
)

print("✅ ChromaDB loaded")


# ============================================================
# 5. GROQ MODEL FOR YOUR RAG
# ============================================================

print("\nCreating Groq model...")

groq_llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
    max_tokens=400,
    api_key=GROQ_API_KEY,
)

print("✅ Groq model created")


# ============================================================
# 6. OPENAI CLIENT POINTING TO GROQ
# ============================================================

print("\nCreating Groq OpenAI-compatible client...")

groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)



print("✅ Groq OpenAI-compatible client created")


# ============================================================
# 7. RAGAS EVALUATOR
# ============================================================

print("\nCreating Ragas evaluator...")

evaluator_llm = llm_factory(
    GROQ_MODEL,
    client=groq_client,
)

print("\nCreating Ragas embeddings...")

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

ragas_embeddings = embedding_factory(
    "openai",
    model="text-embedding-3-small",
    client=openai_client,
    interface="modern",
)

print("✅ Ragas embeddings created")

print("✅ Ragas evaluator created")


# ============================================================
# 8. LOAD TEST DATASET
# ============================================================

print(
    f"\nLoading evaluation dataset: "
    f"{DATASET_PATH}"
)

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"Evaluation dataset not found: "
        f"{DATASET_PATH}"
    )


df = pd.read_csv(DATASET_PATH)


required_columns = [
    "question",
    "reference_answer",
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


print(
    f"✅ Loaded {len(df)} evaluation questions"
)


# ============================================================
# 9. RUN RAG
# ============================================================

print("\n")
print("=" * 70)
print("                    RUNNING RAG")
print("=" * 70)


samples = []


for index, row in df.iterrows():

    question = str(
        row["question"]
    )

    reference_answer = str(
        row["reference_answer"]
    )


    print("\n" + "-" * 70)

    print(
        f"Question {index + 1}/{len(df)}"
    )

    print(
        f"Question: {question}"
    )


    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    docs = vectordb.similarity_search(
        question,
        k=TOP_K,
    )


    # --------------------------------------------------------
    # EXTRACT CONTEXT
    # --------------------------------------------------------

    retrieved_contexts = [
        doc.page_content
        for doc in docs
    ]


    if not retrieved_contexts:

        print(
            "⚠️ No context retrieved"
        )


    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    current_context = "\n\n".join(
        retrieved_contexts
    )


    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    prompt = f"""
You are an academic question-answering assistant.

Use ONLY the retrieved context below to answer
the question.

If the answer cannot be found in the context,
say:

"I don't know based on the uploaded documents."

Do not invent information.

Keep the answer concise.

Retrieved Context:
{current_context}

Question:
{question}

Answer:
"""


    response = groq_llm.invoke(
        prompt
    )

    answer = response.content


    print(
        f"Answer: {answer}"
    )


    # --------------------------------------------------------
    # CREATE RAGAS SAMPLE
    # --------------------------------------------------------

    samples.append(

        {
            "user_input": question,

            "retrieved_contexts":
                retrieved_contexts,

            "response":
                answer,

            "reference":
                reference_answer,
        }
    )


# ============================================================
# 10. CREATE RAGAS DATASET
# ============================================================

print("\nCreating Ragas evaluation dataset...")

evaluation_dataset = (
    EvaluationDataset.from_list(
        samples
    )
)

print(
    "✅ Ragas dataset created"
)


# ============================================================
# 11. RAGAS METRICS
# ============================================================

print("\nPreparing Ragas metrics...")


metrics = [

    # --------------------------------------------------------
    # FAITHFULNESS
    # --------------------------------------------------------

    Faithfulness(
        llm=evaluator_llm
    ),


    # --------------------------------------------------------
    # ANSWER RELEVANCY
    # --------------------------------------------------------

    AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=ragas_embeddings,
    ),

    # --------------------------------------------------------
    # CONTEXT PRECISION
    # --------------------------------------------------------

    ContextPrecision(
        llm=evaluator_llm
    ),


    # --------------------------------------------------------
    # CONTEXT RECALL
    # --------------------------------------------------------

    ContextRecall(
        llm=evaluator_llm
    ),
]


print(
    "✅ Ragas metrics ready"
)


# ============================================================
# 12. RUN EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("                  RUNNING RAGAS")
print("=" * 70)


result = evaluate(
    dataset=evaluation_dataset,
    metrics=metrics,
)


# ============================================================
# 13. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("                    RAGAS RESULTS")
print("=" * 70)

print(result)


# ============================================================
# 14. DATAFRAME
# ============================================================

result_df = result.to_pandas()


print("\n")
print("=" * 70)
print("                 DETAILED RESULTS")
print("=" * 70)


print(
    result_df.to_string(
        index=False
    )
)


# ============================================================
# 15. AVERAGE SCORES
# ============================================================

print("\n")
print("=" * 70)
print("                  AVERAGE SCORES")
print("=" * 70)


metric_columns = [

    "faithfulness",

    "answer_relevancy",

    "context_precision_with_reference",

    "context_recall",
]


summary = []


for metric in metric_columns:

    if metric in result_df.columns:

        score = result_df[
            metric
        ].mean()


        summary.append(
            {
                "metric": metric,
                "average_score": score,
            }
        )


        print(
            f"{metric:<40} "
            f"{score:.4f}"
        )


# ============================================================
# 16. SAVE RESULTS
# ============================================================

os.makedirs(
    "evaluation",
    exist_ok=True
)


result_df.to_csv(
    RESULT_PATH,
    index=False
)


print(
    f"\n✅ Detailed results saved:"
    f"\n   {RESULT_PATH}"
)


# ============================================================
# 17. SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary
)


summary_df.to_csv(
    SUMMARY_PATH,
    index=False
)


print(
    f"✅ Summary saved:"
    f"\n   {SUMMARY_PATH}"
)


# ============================================================
# 18. COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("                 EVALUATION COMPLETE")
print("=" * 70)

print(
    "\nMetrics evaluated:"
)

print("1. Faithfulness")
print("2. Answer Relevancy")
print("3. Context Precision")
print("4. Context Recall")

print(
    "\nOutput:"
)

print(
    f"📄 {RESULT_PATH}"
)

print(
    f"📄 {SUMMARY_PATH}"
)