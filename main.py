import os
import uuid
import shutil
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch,
)

# ---------------------------
# Streamlit config
# ---------------------------
st.set_page_config(page_title="AI PDF Chatbot (RAG)", layout="wide")
st.title("📄 AI PDF Chatbot — Local RAG with Ollama")

# ---------------------------
# Constants
# ---------------------------
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "all-minilm"
OLLAMA_MODEL = "qwen2.5:3b-instruct-q4_K_M"

# ---------------------------
# Cached resources
# ---------------------------
@st.cache_resource
def load_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)

@st.cache_resource
def load_llm():
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.2,
        streaming=True,
    )

@st.cache_resource
def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=load_embeddings(),
    )

# ---------------------------
# PDF loader
# ---------------------------
@st.cache_data
def load_and_split_pdf(file_path: str):
    docs = PyPDFLoader(file_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(docs)

# ---------------------------
# Prompt templates
# ---------------------------
RAG_PROMPT = PromptTemplate(
    template="""
You are an expert AI assistant answering questions from a PDF document.

Use ONLY the provided context.
If the answer is not present, say:
"The document does not contain this information."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)

GREETING_PROMPT = PromptTemplate.from_template(
    "You are a friendly assistant. Reply briefly.\n\nUser: {question}"
)

# ---------------------------
# Greeting detector (cheap router)
# ---------------------------
def is_greeting(text: str) -> bool:
    text = text.lower().strip()
    return any(
        g in text
        for g in ["hi", "hello", "hey", "good morning", "good evening"]
    )

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("📤 Upload PDF")

    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
    )

    clear_db = st.button("🗑️ Clear Vector Database")

if clear_db:
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    st.cache_resource.clear()
    st.cache_data.clear()
    st.success("Vector database cleared.")

# ---------------------------
# Core objects
# ---------------------------
vectorstore = get_vectorstore()
llm = load_llm()

# ---------------------------
# PDF ingestion
# ---------------------------
if uploaded_file:
    temp_path = f"temp_{uuid.uuid4()}.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.status("📄 Processing PDF...", expanded=True) as status:
        status.write("📥 Loading & splitting PDF")
        chunks = load_and_split_pdf(temp_path)

        if chunks:
            status.write("🧠 Storing embeddings")
            vectorstore.add_documents(chunks)
            vectorstore.persist()

        status.update(label="✅ PDF indexed", state="complete")

    os.remove(temp_path)

st.divider()

# ---------------------------
# Chains (THIS is the simplification)
# ---------------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

rag_chain = (
    {
        "context": retriever
        | RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": RunnablePassthrough(),
    }
    | RAG_PROMPT
    | llm
)

greeting_chain = (
    {"question": RunnablePassthrough()}
    | GREETING_PROMPT
    | llm
)

main_chain = RunnableBranch(
    (lambda x: is_greeting(x), greeting_chain),
    rag_chain,  # default path
)

# ---------------------------
# Chat UI
# ---------------------------
st.subheader("💬 Ask questions")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input(
    "Ask a question about the document",
    placeholder="What is this document about?",
)

if st.button("Ask") and question:
    answer_placeholder = st.empty()
    streamed_answer = ""

    with st.status("⚙️ Processing...", expanded=True) as status:
        status.write("🧠 UnderStanding question")

        for chunk in main_chain.stream(question):
            if chunk.content:
                streamed_answer += chunk.content
                answer_placeholder.markdown(streamed_answer + "▌")

        status.update(label="✅ Done", state="complete")

    st.session_state.chat_history.append((question, streamed_answer))

# ---------------------------
# Chat history
# ---------------------------
for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"**🧑 Question:** {q}")
    st.markdown(f"**🤖 Answer:** {a}")
    st.divider()
