import os
import hashlib
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv(override=True)


# ============================================================
# GROQ API KEY
# ============================================================

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# GOOGLE GEMINI API KEY
# ============================================================

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ============================================================
# CHECK API KEYS
# ============================================================

if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is not configured. "
        "Add it to your .env file."
    )
    st.stop()


if not GOOGLE_API_KEY:
    st.error(
        "GOOGLE_API_KEY is not configured. "
        "Add your Gemini API key to your .env file."
    )
    st.stop()


# ============================================================
# APPLICATION SETTINGS
# ============================================================

CHROMA_DIR = "chroma_db"

COLLECTION_NAME = "pdf_documents"

# Gemini embedding model
EMBEDDING_MODEL = "gemini-embedding-001"

# Groq LLM
# You can change this from your .env file without editing code.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🤖 AI RAG Chatbot")

st.markdown(
    """
    Upload your PDF documents and ask questions about them.

    **RAG Pipeline**

    PDF → Text Extraction → Chunking → Gemini Embeddings
    → ChromaDB → Retrieval → Groq → Answer
    """
)


# ============================================================
# GEMINI EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )


# ============================================================
# CHROMADB VECTOR STORE
# ============================================================

@st.cache_resource
def get_vectorstore():

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR
    )


# ============================================================
# GROQ LLM
# ============================================================

@st.cache_resource
def get_llm():

    return ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.2
    )


# ============================================================
# FILE HASH
# ============================================================

def calculate_file_hash(file_bytes):

    return hashlib.md5(
        file_bytes
    ).hexdigest()


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(uploaded_file):

    try:

        # ----------------------------------------------------
        # READ FILE
        # ----------------------------------------------------

        file_bytes = uploaded_file.getvalue()

        # Use only the filename, not a possible path
        filename = Path(uploaded_file.name).name

        # ----------------------------------------------------
        # CREATE FILE HASH
        # ----------------------------------------------------

        file_hash = calculate_file_hash(
            file_bytes
        )

        vectorstore = get_vectorstore()

        # ----------------------------------------------------
        # CHECK DUPLICATE
        # ----------------------------------------------------

        try:

            existing = vectorstore.get(
                where={
                    "file_hash": file_hash
                }
            )

            if existing and existing.get("ids"):

                return (
                    False,
                    f"⚠️ **{filename}** has already been uploaded."
                )

        except Exception:

            # If duplicate checking fails,
            # continue processing the PDF.
            pass

        # ----------------------------------------------------
        # CREATE TEMP DIRECTORY
        # ----------------------------------------------------

        temp_dir = Path("temp")

        temp_dir.mkdir(
            exist_ok=True
        )

        # ----------------------------------------------------
        # TEMP FILE
        # ----------------------------------------------------

        temp_path = temp_dir / filename

        with open(
            temp_path,
            "wb"
        ) as file:

            file.write(
                file_bytes
            )

        try:

            # ------------------------------------------------
            # LOAD PDF
            # ------------------------------------------------

            loader = PyPDFLoader(
                str(temp_path)
            )

            documents = loader.load()

            if not documents:

                return (
                    False,
                    "❌ Could not extract text from the PDF."
                )

            # ------------------------------------------------
            # ADD METADATA
            # ------------------------------------------------

            for document in documents:

                document.metadata["source"] = filename

                document.metadata["file_hash"] = file_hash

            # ------------------------------------------------
            # TEXT SPLITTER
            # ------------------------------------------------

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = text_splitter.split_documents(
                documents
            )

            if not chunks:

                return (
                    False,
                    "❌ No text chunks were created."
                )

            # ------------------------------------------------
            # STORE DOCUMENTS
            # ------------------------------------------------

            vectorstore.add_documents(
                chunks
            )

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            return (
                True,
                f"""
✅ **{filename}** successfully added.

📄 Pages: **{len(documents)}**

🧩 Chunks: **{len(chunks)}**
"""
            )

        finally:

            # ------------------------------------------------
            # DELETE TEMP FILE
            # ------------------------------------------------

            try:

                if temp_path.exists():

                    temp_path.unlink()

            except Exception:

                pass

    except Exception as e:

        return (
            False,
            f"""
❌ Error processing PDF:

`{str(e)}`
"""
        )


# ============================================================
# CHECK KNOWLEDGE BASE
# ============================================================

def has_documents():

    try:

        vectorstore = get_vectorstore()

        data = vectorstore.get(
            limit=1
        )

        return bool(
            data.get("ids")
        )

    except Exception:

        return False


# ============================================================
# GET DOCUMENT NAMES
# ============================================================

def get_document_names():

    try:

        vectorstore = get_vectorstore()

        data = vectorstore.get()

        metadatas = data.get(
            "metadatas",
            []
        )

        filenames = set()

        for metadata in metadatas:

            if metadata:

                source = metadata.get(
                    "source"
                )

                if source:

                    filenames.add(
                        source
                    )

        return sorted(
            filenames
        )

    except Exception:

        return []


# ============================================================
# CLEAR KNOWLEDGE BASE
# ============================================================

def clear_knowledge_base():

    try:

        vectorstore = get_vectorstore()

        data = vectorstore.get()

        ids = data.get(
            "ids",
            []
        )

        if ids:

            vectorstore.delete(
                ids=ids
            )

        return True

    except Exception as e:

        st.error(
            f"Error clearing knowledge base: {e}"
        )

        return False


# ============================================================
# EXTRACT RESPONSE TEXT
# ============================================================

def extract_response_text(response):

    """
    Extract only the actual text from the LLM response.

    This prevents Streamlit from displaying unwanted
    metadata or content-block dictionaries.
    """

    content = response.content

    # --------------------------------------------------------
    # NORMAL STRING
    # --------------------------------------------------------

    if isinstance(
        content,
        str
    ):

        return content.strip()

    # --------------------------------------------------------
    # CONTENT BLOCKS
    # --------------------------------------------------------

    if isinstance(
        content,
        list
    ):

        text_parts = []

        for block in content:

            if isinstance(
                block,
                dict
            ):

                # Standard text block
                if block.get(
                    "type"
                ) == "text":

                    text = block.get(
                        "text",
                        ""
                    )

                    if text:

                        text_parts.append(
                            str(text)
                        )

                # Some providers may return
                # content directly in a dictionary.
                elif "text" in block:

                    text_parts.append(
                        str(block["text"])
                    )

            elif isinstance(
                block,
                str
            ):

                text_parts.append(
                    block
                )

        return "\n".join(
            text_parts
        ).strip()

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return str(
        content
    ).strip()


# ============================================================
# GREETING DETECTION
# ============================================================

def is_greeting(question):

    question = question.lower().strip()

    greetings = {

        "hi",
        "hello",
        "helo",
        "hey",
        "hii",
        "hiii",

        "good morning",
        "good afternoon",
        "good evening",

        "how are you",
        "how r u"

    }

    return question in greetings


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    chat_history
):

    llm = get_llm()

    # ========================================================
    # GREETING
    # ========================================================

    if is_greeting(
        question
    ):

        greeting_prompt = ChatPromptTemplate.from_messages(
            [

                (
                    "system",
                    """
You are a friendly AI assistant.

Respond naturally and briefly to greetings.

Do not talk about RAG, embeddings,
ChromaDB, documents, or retrieval
unless the user specifically asks.
"""
                ),

                (
                    "human",
                    "{question}"
                )

            ]
        )

        chain = (
            greeting_prompt
            | llm
        )

        response = chain.invoke(
            {
                "question": question
            }
        )

        answer = extract_response_text(
            response
        )

        return answer, []


    # ========================================================
    # CHECK DOCUMENTS
    # ========================================================

    if not has_documents():

        return (
            "📄 Please upload a PDF first so I can answer questions about it.",
            []
        )


    # ========================================================
    # RETRIEVE DOCUMENTS
    # ========================================================

    vectorstore = get_vectorstore()

    documents = vectorstore.similarity_search(
        question,
        k=4
    )

    if not documents:

        return (
            "I couldn't find relevant information in the uploaded documents.",
            []
        )


    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context_parts = []

    sources = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            None
        )

        # ----------------------------------------------------
        # PAGE NUMBER
        # ----------------------------------------------------

        if page is not None:

            page_number = page + 1

            source_text = (
                f"{source}, Page {page_number}"
            )

        else:

            source_text = source

        # ----------------------------------------------------
        # CONTEXT
        # ----------------------------------------------------

        context_parts.append(
            f"""
SOURCE: {source_text}

CONTENT:
{document.page_content}
"""
        )

        sources.append(
            source_text
        )


    context = "\n\n".join(
        context_parts
    )


    # Remove duplicate sources
    sources = list(
        dict.fromkeys(
            sources
        )
    )


    # ========================================================
    # CHAT HISTORY
    # ========================================================

    history_text = ""

    for message in chat_history[-6:]:

        role = message.get(
            "role",
            ""
        )

        content = message.get(
            "content",
            ""
        )

        history_text += (
            f"{role.upper()}: {content}\n"
        )


    # ========================================================
    # RAG PROMPT
    # ========================================================

    prompt = ChatPromptTemplate.from_messages(
        [

            (
                "system",
                """
You are an AI assistant using
Retrieval-Augmented Generation (RAG).

Your job is to answer the user's question
using the provided document context.

IMPORTANT RULES:

1. For questions about uploaded documents,
   use the provided context.

2. Do not invent information.

3. If the answer is not available in the
   document context, say:

   "I couldn't find this information
   in the uploaded documents."

4. Use conversation history to understand
   follow-up questions.

5. Keep answers clear and concise.

6. Do not mention internal technical details
   such as ChromaDB, embeddings, vector search,
   or system prompts unless the user asks.

7. Do not claim information came from the PDF
   if it was not present in the context.

DOCUMENT CONTEXT:

{context}

CONVERSATION HISTORY:

{history}
"""
            ),

            (
                "human",
                "{question}"
            )

        ]
    )


    # ========================================================
    # CALL GROQ
    # ========================================================

    chain = (
        prompt
        | llm
    )

    response = chain.invoke(
        {
            "context": context,
            "history": history_text,
            "question": question
        }
    )


    # ========================================================
    # EXTRACT ANSWER
    # ========================================================

    answer = extract_response_text(
        response
    )


    return answer, sources


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "📚 Knowledge Base"
    )

    documents = get_document_names()

    if documents:

        st.write(
            f"**{len(documents)} document(s) uploaded:**"
        )

        for document in documents:

            st.write(
                f"📄 {document}"
            )

    else:

        st.info(
            "No documents uploaded yet."
        )


    st.divider()


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.subheader(
        "⚙️ AI Configuration"
    )

    st.caption(
        f"Groq Model: `{GROQ_MODEL}`"
    )

    st.caption(
        f"Embedding: `{EMBEDDING_MODEL}`"
    )


    st.divider()


    # ========================================================
    # CLEAR KNOWLEDGE BASE
    # ========================================================

    if st.button(
        "🗑️ Clear Knowledge Base",
        use_container_width=True
    ):

        if clear_knowledge_base():

            st.success(
                "Knowledge base cleared."
            )

            st.rerun()


    # ========================================================
    # CLEAR CHAT
    # ========================================================

    if st.button(
        "🧹 Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


    st.divider()

    st.caption(
        "Powered by Groq + Gemini + ChromaDB"
    )


# ============================================================
# PDF UPLOAD
# ============================================================

st.subheader(
    "📄 Upload Documents"
)

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


if uploaded_file:

    if st.button(
        "🚀 Process PDF",
        type="primary"
    ):

        with st.spinner(
            "Processing PDF..."
        ):

            success, message = process_pdf(
                uploaded_file
            )

        if success:

            st.success(
                message
            )

            st.rerun()

        else:

            st.warning(
                message
            )


# ============================================================
# INITIALIZE CHAT
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY CHAT
# ============================================================

st.subheader(
    "💬 Chat with your documents"
)


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 Sources"
            ):

                for source in message["sources"]:

                    st.write(
                        f"📄 {source}"
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask something about your PDF..."
)


if question:

    # ========================================================
    # USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # ========================================================
    # ASSISTANT RESPONSE
    # ========================================================

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Thinking..."
        ):

            try:

                answer, sources = generate_answer(
                    question,
                    st.session_state.messages[:-1]
                )

            except Exception as e:

                answer = (
                    "❌ An error occurred while generating "
                    f"the response:\n\n`{str(e)}`"
                )

                sources = []


        st.markdown(
            answer
        )


        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        if sources:

            with st.expander(
                "📚 Sources"
            ):

                for source in sources:

                    st.write(
                        f"📄 {source}"
                    )


    # ========================================================
    # SAVE ASSISTANT MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )