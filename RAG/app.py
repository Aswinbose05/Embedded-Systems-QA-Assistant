import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="Embedded Systems QA Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Embedded Systems QA Assistant")
st.caption("Upload your document and ask questions using RAG + Groq")


# ============================================================
# GROQ API KEY
# ============================================================

groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit Cloud Secrets fallback
if not groq_api_key:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_api_key = None

if not groq_api_key:
    st.error(
        "GROQ_API_KEY is missing. "
        "Add it to your .env file locally or Streamlit Secrets when deployed."
    )
    st.stop()


# ============================================================
# SETTINGS
# ============================================================

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "embedded_systems_qa"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K = 4


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


embeddings = load_embeddings()


# ============================================================
# LOAD GROQ MODEL
# ============================================================

@st.cache_resource
def load_llm():

    return ChatGroq(
        api_key=groq_api_key,
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_tokens=500,
    )


llm = load_llm()


# ============================================================
# LOAD DOCUMENT
# ============================================================

def load_document(file_path, file_name):

    extension = Path(file_name).suffix.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(file_path)

    elif extension == ".txt":
        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )

    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)

    else:
        raise ValueError(
            "Unsupported file type. Please upload PDF, TXT or DOCX."
        )

    return loader.load()


# ============================================================
# SPLIT DOCUMENT
# ============================================================

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    return splitter.split_documents(documents)


# ============================================================
# CREATE CHROMA VECTOR DATABASE
# ============================================================

def create_vectorstore(chunks):

    # Remove old database
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    return vectorstore


# ============================================================
# PROCESS UPLOADED DOCUMENT
# ============================================================

def process_document(uploaded_file):

    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:

        documents = load_document(
            temp_path,
            uploaded_file.name
        )

        if not documents:
            raise ValueError(
                "No text could be extracted from this document."
            )

        chunks = split_documents(documents)

        if not chunks:
            raise ValueError(
                "Document was loaded but no chunks were created."
            )

        vectorstore = create_vectorstore(chunks)

        return vectorstore, documents, chunks

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, retrieved_docs):

    if not retrieved_docs:

        return (
            "I don't know based on the uploaded documents."
        )

    context_parts = []

    for i, (doc, score) in enumerate(retrieved_docs, start=1):

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            None
        )

        if page is not None:
            page = page + 1

        context_parts.append(
            f"""
DOCUMENT CHUNK {i}
Source: {source}
Page: {page if page else "N/A"}

Content:
{doc.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an AI assistant for question answering over uploaded documents.

Use ONLY the information provided in the document context.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Instructions:

1. Answer the question using the document context.
2. Give a clear and useful answer.
3. Do not simply copy the entire document.
4. If the context contains the answer, explain it naturally.
5. If the answer is not present in the context, say:
   "I don't know based on the uploaded documents."
6. Do not invent information.

ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Document Upload")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "docx"],
    )

    process_button = st.button(
        "⚙️ Process Document",
        use_container_width=True,
    )

    st.divider()

    st.subheader("⚙️ RAG Settings")

    st.write(
        f"**Embedding:** `{EMBEDDING_MODEL}`"
    )

    st.write(
        f"**Chunk Size:** `{CHUNK_SIZE}`"
    )

    st.write(
        f"**Top K:** `{TOP_K}`"
    )

    st.write(
        "**LLM:** `llama-3.1-8b-instant`"
    )


# ============================================================
# PROCESS DOCUMENT BUTTON
# ============================================================

if process_button:

    if uploaded_file is None:

        st.warning(
            "Please upload a document first."
        )

    else:

        with st.spinner(
            "Processing document..."
        ):

            try:

                vectorstore, documents, chunks = (
                    process_document(uploaded_file)
                )

                st.session_state.vectorstore = vectorstore
                st.session_state.document_name = uploaded_file.name
                st.session_state.chunk_count = len(chunks)

                st.session_state.chat_history = []

                st.success(
                    f"✅ {uploaded_file.name} processed successfully!"
                )

                st.info(
                    f"📄 Pages/Documents: {len(documents)}  |  "
                    f"🧩 Chunks: {len(chunks)}"
                )

            except Exception as e:

                st.error(
                    f"❌ Error processing document: {e}"
                )


# ============================================================
# DOCUMENT STATUS
# ============================================================

if "vectorstore" in st.session_state:

    st.success(
        f"📚 Active document: "
        f"**{st.session_state.document_name}**"
    )

    st.caption(
        f"Created {st.session_state.chunk_count} chunks"
    )


# ============================================================
# CHAT HISTORY
# ============================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your uploaded document..."
)


if question:

    # --------------------------------------------
    # Check document
    # --------------------------------------------

    if "vectorstore" not in st.session_state:

        st.warning(
            "Please upload and process a document first."
        )

        st.stop()


    # --------------------------------------------
    # Display question
    # --------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------
    # Retrieve + Generate
    # --------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Retrieving information and generating answer..."
        ):

            try:

                vectorstore = (
                    st.session_state.vectorstore
                )

                # Retrieve relevant chunks
                retrieved_docs = (
                    vectorstore.similarity_search_with_score(
                        question,
                        k=TOP_K
                    )
                )

                # --------------------------------------------
                # DEBUG: SHOW RETRIEVED DOCUMENTS
                # --------------------------------------------

                with st.expander(
                    "🔎 Retrieved document chunks"
                ):

                    if not retrieved_docs:

                        st.warning(
                            "No relevant chunks were retrieved."
                        )

                    else:

                        for i, (doc, score) in enumerate(
                            retrieved_docs,
                            start=1
                        ):

                            source = doc.metadata.get(
                                "source",
                                "Unknown"
                            )

                            page = doc.metadata.get(
                                "page",
                                None
                            )

                            if page is not None:
                                page = page + 1

                            st.markdown(
                                f"### Chunk {i}"
                            )

                            st.write(
                                f"**Source:** {source}"
                            )

                            st.write(
                                f"**Page:** "
                                f"{page if page else 'N/A'}"
                            )

                            st.write(
                                f"**Similarity score:** "
                                f"{score:.4f}"
                            )

                            st.write(
                                doc.page_content
                            )

                            st.divider()


                # --------------------------------------------
                # GENERATE ANSWER WITH GROQ
                # --------------------------------------------

                answer = generate_answer(
                    question,
                    retrieved_docs
                )


                # --------------------------------------------
                # DISPLAY ANSWER
                # --------------------------------------------

                st.markdown("### 🤖 Answer")

                st.markdown(answer)


                # --------------------------------------------
                # SOURCE INFORMATION
                # --------------------------------------------

                if retrieved_docs:

                    sources = []

                    for doc, score in retrieved_docs:

                        source = doc.metadata.get(
                            "source",
                            "Unknown"
                        )

                        page = doc.metadata.get(
                            "page",
                            None
                        )

                        if page is not None:
                            page = page + 1

                        source_info = (
                            f"{Path(source).name}"
                        )

                        if page:
                            source_info += (
                                f" — Page {page}"
                            )

                        if source_info not in sources:
                            sources.append(
                                source_info
                            )

                    with st.expander(
                        "📚 Sources"
                    ):

                        for source in sources:

                            st.write(
                                f"- {source}"
                            )


                # --------------------------------------------
                # SAVE ANSWER
                # --------------------------------------------

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )


            except Exception as e:

                error_message = (
                    f"❌ Error generating answer: {e}"
                )

                st.error(error_message)

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )