import os
import hashlib
import pandas as pd
import streamlit as st

from dotenv import load_dotenv

# LangChain
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader
)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_groq import ChatGroq

# LangGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        GROQ_API_KEY = None

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not configured.")
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# =========================================================
# CONFIGURATION
# =========================================================

CHROMA_DIRECTORY = "./chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K = 3


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Academic QA",
    page_icon="💬",
    layout="centered"
)


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def get_embeddings_model():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


embeddings = get_embeddings_model()


# =========================================================
# CHROMA VECTOR DATABASE
# =========================================================

@st.cache_resource
def get_vector_store():

    return Chroma(
        collection_name="academic_qa",
        persist_directory=CHROMA_DIRECTORY,
        embedding_function=embeddings
    )


vectordb = get_vector_store()


# =========================================================
# GROQ MODEL
# =========================================================

@st.cache_resource
def get_chat_model():

    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=400
    )


model = get_chat_model()


# =========================================================
# DOCUMENT HASH
# =========================================================

def get_file_hash(uploaded_file):

    return hashlib.md5(
        uploaded_file.getvalue()
    ).hexdigest()


# =========================================================
# SAVE UPLOADED FILE
# =========================================================

def save_uploaded_file(uploaded_file):

    upload_directory = "./uploaded_documents"

    os.makedirs(upload_directory, exist_ok=True)

    file_path = os.path.join(
        upload_directory,
        uploaded_file.name
    )

    with open(file_path, "wb") as f:

        f.write(uploaded_file.getbuffer())

    return file_path


# =========================================================
# LOAD DOCUMENT
# =========================================================

def load_document(file_path):

    extension = file_path.lower().split(".")[-1]

    if extension == "pdf":

        loader = PyPDFLoader(file_path)

    elif extension == "docx":

        loader = Docx2txtLoader(file_path)

    elif extension == "txt":

        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )

    elif extension == "csv":

        loader = CSVLoader(file_path)

    else:

        raise ValueError(
            f"Unsupported file format: .{extension}"
        )

    documents = loader.load()

    return documents


# =========================================================
# PROCESS DOCUMENT
# =========================================================

def process_document(uploaded_file):

    # Save document
    file_path = save_uploaded_file(uploaded_file)

    # Load document
    documents = load_document(file_path)

    # Add source metadata
    for document in documents:

        document.metadata["source"] = uploaded_file.name

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    # Add unique document ID
    for i, chunk in enumerate(chunks):

        chunk.metadata["document_id"] = get_file_hash(
            uploaded_file
        )

        chunk.metadata["chunk_id"] = i

    # Store in Chroma
    vectordb.add_documents(chunks)

    return len(chunks)


# =========================================================
# LANGGRAPH MODEL NODE
# =========================================================

def call_model(state: MessagesState):

    system_prompt = """
You are an academic question-answering assistant.

Answer the user's question using ONLY the retrieved context.

Rules:
1. If the answer is available in the context, answer clearly.
2. If the answer is not available in the context, say:
   "I don't know based on the uploaded documents."
3. Do not invent information.
4. Keep the answer concise.
5. Maximum 3 sentences unless explanation is necessary.
"""

    messages = [
        SystemMessage(content=system_prompt)
    ] + state["messages"]

    response = model.invoke(messages)

    return {
        "messages": [response]
    }


# =========================================================
# LANGGRAPH
# =========================================================

@st.cache_resource
def get_langgraph_app():

    workflow = StateGraph(
        state_schema=MessagesState
    )

    workflow.add_node(
        "model",
        call_model
    )

    workflow.add_edge(
        START,
        "model"
    )

    memory = MemorySaver()

    app = workflow.compile(
        checkpointer=memory
    )

    return app


app = get_langgraph_app()


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "thread_id" not in st.session_state:

    st.session_state.thread_id = "streamlit_chat_session"


if "processed_files" not in st.session_state:

    st.session_state.processed_files = set()


# =========================================================
# UI
# =========================================================

st.title("💬 Academic QA")

st.caption(
    "Upload your documents and ask questions using RAG."
)


# =========================================================
# SIDEBAR - DOCUMENT UPLOAD
# =========================================================

with st.sidebar:

    st.header("📚 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[
            "pdf",
            "docx",
            "txt",
            "csv"
        ],
        accept_multiple_files=True
    )

    process_button = st.button(
        "⚙️ Process Documents",
        use_container_width=True
    )


    if process_button:

        if not uploaded_files:

            st.warning(
                "Please upload at least one document."
            )

        else:

            progress = st.progress(0)

            for index, uploaded_file in enumerate(
                uploaded_files
            ):

                file_hash = get_file_hash(
                    uploaded_file
                )

                # Avoid processing same file twice
                if file_hash in st.session_state.processed_files:

                    st.info(
                        f"Already processed: {uploaded_file.name}"
                    )

                    continue

                try:

                    with st.spinner(
                        f"Processing {uploaded_file.name}..."
                    ):

                        chunks_count = process_document(
                            uploaded_file
                        )

                    st.session_state.processed_files.add(
                        file_hash
                    )

                    st.success(
                        f"✅ {uploaded_file.name} processed "
                        f"({chunks_count} chunks)"
                    )

                except Exception as e:

                    st.error(
                        f"❌ Error processing "
                        f"{uploaded_file.name}: {e}"
                    )

                progress.progress(
                    (index + 1) / len(uploaded_files)
                )


# =========================================================
# SHOW CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

if prompt := st.chat_input(
    "Ask a question about your documents..."
):

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)


    # =====================================================
    # ASSISTANT
    # =====================================================

    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):

            try:

                # -----------------------------------------
                # RETRIEVE DOCUMENTS
                # -----------------------------------------

                docs = vectordb.similarity_search_with_score(
                    prompt,
                    k=TOP_K
                )


                # -----------------------------------------
                # CHECK WHETHER DOCUMENTS EXIST
                # -----------------------------------------

                if not docs:

                    answer = (
                        "I don't know based on the "
                        "uploaded documents."
                    )

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                    st.stop()


                # -----------------------------------------
                # CREATE CONTEXT
                # -----------------------------------------

                context_parts = []

                sources = []

                for doc, score in docs:

                    source = doc.metadata.get(
                        "source",
                        "Unknown"
                    )

                    page = doc.metadata.get(
                        "page",
                        None
                    )

                    if page is not None:

                        page_number = int(page) + 1

                    else:

                        page_number = "N/A"


                    context_parts.append(
                        f"""
Source: {source}
Page: {page_number}

Content:
{doc.page_content}
"""
                    )


                    sources.append(
                        {
                            "source": source,
                            "page": page_number,
                            "score": score
                        }
                    )


                current_context = "\n\n".join(
                    context_parts
                )


                # -----------------------------------------
                # CREATE HUMAN MESSAGE
                # -----------------------------------------

                current_turn_message = HumanMessage(
                    content=f"""
Retrieved Context:

{current_context}

Question:

{prompt}
"""
                )


                # -----------------------------------------
                # LANGGRAPH
                # -----------------------------------------

                result = app.invoke(
                    {
                        "messages": [
                            current_turn_message
                        ]
                    },
                    config={
                        "configurable": {
                            "thread_id":
                            st.session_state.thread_id
                        }
                    }
                )


                # -----------------------------------------
                # GET RESPONSE
                # -----------------------------------------

                ai_response = result[
                    "messages"
                ][-1].content


                # -----------------------------------------
                # SOURCE INFORMATION
                # -----------------------------------------

                source_text = "\n\n### 📚 Sources\n"

                seen_sources = set()

                for source in sources:

                    key = (
                        source["source"],
                        source["page"]
                    )

                    if key in seen_sources:

                        continue

                    seen_sources.add(key)

                    source_text += (
                        f"- **{source['source']}** "
                        f"(Page {source['page']})\n"
                    )


                # -----------------------------------------
                # FINAL RESPONSE
                # -----------------------------------------

                final_response = (
                    f"{ai_response}"
                    f"{source_text}"
                )


                st.markdown(
                    final_response
                )


                # -----------------------------------------
                # SAVE CHAT
                # -----------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": final_response
                    }
                )


            except Exception as e:

                error_message = (
                    f"❌ An error occurred: {str(e)}"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )