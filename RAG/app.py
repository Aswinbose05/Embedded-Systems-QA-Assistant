import os
import hashlib
import streamlit as st

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        GROQ_API_KEY = None

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is not configured.")
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# =========================================================
# CONFIGURATION
# =========================================================

CHROMA_DIRECTORY = "./chroma_db"

COLLECTION_NAME = "academic_qa"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

TOP_K = 5


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Academic QA",
    page_icon="🤖",
    layout="wide",
)


# =========================================================
# EMBEDDINGS
# =========================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


embeddings = get_embeddings()


# =========================================================
# CHROMA
# =========================================================

@st.cache_resource
def get_vector_store():

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIRECTORY,
        embedding_function=embeddings,
    )


vectordb = get_vector_store()


# =========================================================
# GROQ
# =========================================================

@st.cache_resource
def get_llm():

    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=500,
    )


llm = get_llm()


# =========================================================
# FILE HASH
# =========================================================

def get_file_hash(uploaded_file):

    return hashlib.md5(
        uploaded_file.getvalue()
    ).hexdigest()


# =========================================================
# LOAD DOCUMENT
# =========================================================

def load_document(uploaded_file):

    extension = uploaded_file.name.lower().split(".")[-1]

    temp_directory = "./uploaded_documents"

    os.makedirs(
        temp_directory,
        exist_ok=True
    )

    file_path = os.path.join(
        temp_directory,
        uploaded_file.name
    )

    with open(file_path, "wb") as file:

        file.write(
            uploaded_file.getbuffer()
        )


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
            f"Unsupported file type: {extension}"
        )


    documents = loader.load()


    # Add source information

    for document in documents:

        document.metadata["source"] = uploaded_file.name


    return documents


# =========================================================
# PROCESS DOCUMENT
# =========================================================

def process_document(uploaded_file):

    documents = load_document(
        uploaded_file
    )


    # Split

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )


    chunks = splitter.split_documents(
        documents
    )


    document_id = get_file_hash(
        uploaded_file
    )


    # Add metadata

    for index, chunk in enumerate(chunks):

        chunk.metadata["document_id"] = document_id

        chunk.metadata["chunk_id"] = index


    # Unique IDs for Chroma

    ids = []

    for index in range(len(chunks)):

        ids.append(
            f"{document_id}_{index}"
        )


    # Add documents

    vectordb.add_documents(
        documents=chunks,
        ids=ids,
    )


    return len(chunks)


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(
    question,
    documents
):

    context_parts = []


    for index, document in enumerate(documents):

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            None
        )


        if page is not None:

            page_number = int(page) + 1

        else:

            page_number = "N/A"


        context_parts.append(
            f"""
DOCUMENT {index + 1}

Source: {source}

Page: {page_number}

Content:
{document.page_content}
"""
        )


    context = "\n\n".join(
        context_parts
    )


    prompt = f"""
You are an intelligent document question-answering assistant.

Your job is to answer the user's question using the provided document context.

IMPORTANT RULES:

1. Read the entire provided context carefully.
2. Answer using information from the context.
3. If the answer is clearly present, explain it naturally.
4. You may combine information from multiple retrieved chunks.
5. Do not say "I don't know" if the answer can reasonably be found in the context.
6. Do not use outside knowledge.
7. If the answer genuinely does not exist in the context, say:
   "I couldn't find the answer in the uploaded documents."
8. Give a clear and useful answer.
9. Use simple language.
10. If appropriate, use bullet points or examples.

DOCUMENT CONTEXT:
=================

{context}

=================

USER QUESTION:

{question}

ANSWER:
"""


    response = llm.invoke(
        prompt
    )


    return response.content


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "processed_files" not in st.session_state:

    st.session_state.processed_files = set()


# =========================================================
# HEADER
# =========================================================

st.title(
    "🤖 Academic QA"
)

st.caption(
    "Upload your documents and ask questions using RAG."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "📚 Knowledge Base"
    )


    uploaded_files = st.file_uploader(

        "Upload documents",

        type=[
            "pdf",
            "docx",
            "txt",
            "csv",
        ],

        accept_multiple_files=True,
    )


    process_button = st.button(

        "🚀 Process Documents",

        use_container_width=True,
    )


    # =====================================================
    # PROCESS BUTTON
    # =====================================================

    if process_button:

        if not uploaded_files:

            st.warning(
                "Please upload at least one document."
            )

        else:

            progress = st.progress(0)

            processed_count = 0


            for index, uploaded_file in enumerate(
                uploaded_files
            ):

                file_hash = get_file_hash(
                    uploaded_file
                )


                # Avoid duplicate processing

                if file_hash in st.session_state.processed_files:

                    st.info(
                        f"Already processed: "
                        f"{uploaded_file.name}"
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


                    processed_count += 1


                    st.success(
                        f"✅ {uploaded_file.name} "
                        f"processed successfully "
                        f"({chunks_count} chunks)"
                    )


                except Exception as error:

                    st.error(
                        f"❌ Error processing "
                        f"{uploaded_file.name}: "
                        f"{error}"
                    )


                progress.progress(
                    (index + 1) / len(uploaded_files)
                )


            if processed_count > 0:

                st.success(
                    "🎉 Documents added to the knowledge base!"
                )


    # =====================================================
    # KNOWLEDGE BASE INFO
    # =====================================================

    st.divider()

    st.subheader(
        "🔎 Knowledge Base"
    )


    try:

        collection_data = vectordb.get()

        total_chunks = len(
            collection_data["ids"]
        )

    except Exception:

        total_chunks = 0


    st.metric(
        "Stored Chunks",
        total_chunks
    )


# =========================================================
# CHAT HISTORY
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

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    # =====================================================
    # USER
    # =====================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )


    with st.chat_message("user"):

        st.markdown(question)


    # =====================================================
    # ASSISTANT
    # =====================================================

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "🔎 Searching documents..."
            ):

                # Retrieve documents

                results = vectordb.similarity_search_with_score(

                    question,

                    k=TOP_K,
                )


            if not results:

                answer = (
                    "I couldn't find any relevant "
                    "information in the uploaded documents."
                )

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                st.stop()


            # =================================================
            # GET DOCUMENTS
            # =================================================

            documents = [
                document
                for document, score in results
            ]


            # =================================================
            # GENERATE ANSWER
            # =================================================

            with st.spinner(
                "🤖 Generating answer..."
            ):

                answer = generate_answer(
                    question,
                    documents,
                )


            # =================================================
            # DISPLAY ANSWER
            # =================================================

            st.markdown(
                "### 🤖 Answer"
            )

            st.markdown(
                answer
            )


            # =================================================
            # SOURCES
            # =================================================

            st.markdown(
                "### 📚 Sources"
            )


            seen_sources = set()


            for document, score in results:

                source = document.metadata.get(
                    "source",
                    "Unknown"
                )

                page = document.metadata.get(
                    "page",
                    None
                )


                if page is not None:

                    page_number = int(page) + 1

                else:

                    page_number = "N/A"


                source_key = (
                    source,
                    page_number,
                )


                if source_key in seen_sources:

                    continue


                seen_sources.add(
                    source_key
                )


                st.write(
                    f"📄 **{source}** "
                    f"— Page {page_number}"
                )


            # =================================================
            # SAVE CHAT
            # =================================================

            final_response = (
                f"{answer}\n\n"
                f"### 📚 Sources\n"
            )


            for source, page in seen_sources:

                final_response += (
                    f"- **{source}** "
                    f"(Page {page})\n"
                )


            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_response,
                }
            )


        except Exception as error:

            error_message = (
                f"❌ Error generating answer: "
                f"{error}"
            )


            st.error(
                error_message
            )


            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )