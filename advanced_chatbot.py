import os
import uuid
import shutil
import streamlit as st
import numpy as np

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain_core.prompts import PromptTemplate
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

@st.cache_resource
def get_intent_embeddings():
    """Pre-compute embeddings for intent classification (much faster than LLM calls)"""
    embeddings = load_embeddings()
    
    # Example phrases for each intent
    greeting_examples = [
        "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
        "how are you", "what's up", "howdy", "greetings", "hi there",
        "hello there", "hey there", "bonjour", "hola", "namaste"
    ]
    
    document_examples = [
        "what does the document say about",
        "can you explain this section",
        "what is mentioned about",
        "find information on",
        "tell me about the content",
        "summarize the document",
        "what are the key points"
    ]
    
    # Compute embeddings
    greeting_vecs = [embeddings.embed_query(ex) for ex in greeting_examples]
    document_vecs = [embeddings.embed_query(ex) for ex in document_examples]
    
    return {
        "greeting": np.array(greeting_vecs),
        "document": np.array(document_vecs),
        "embeddings": embeddings
    }

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
# Semantic intent detection (FAST!)
# ---------------------------
def cosine_similarity(a, b):
    """Compute cosine similarity between vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def classify_intent_semantic(text: str) -> bool:
    """
    Use semantic similarity with example embeddings for FAST intent classification.
    This is much faster than calling the LLM and works across all languages.
    """
    intent_data = get_intent_embeddings()
    
    # Embed the user's question
    query_vec = np.array(intent_data["embeddings"].embed_query(text))
    
    # Calculate average similarity to greeting examples
    greeting_sims = [
        cosine_similarity(query_vec, gvec) 
        for gvec in intent_data["greeting"]
    ]
    avg_greeting_sim = np.mean(greeting_sims)
    
    # Calculate average similarity to document question examples
    document_sims = [
        cosine_similarity(query_vec, dvec) 
        for dvec in intent_data["document"]
    ]
    avg_document_sim = np.mean(document_sims)
    
    # Return True if more similar to greetings (for RunnableBranch)
    return avg_greeting_sim > avg_document_sim

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("📤 Upload PDF")

    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
    )
    
    st.divider()
    st.subheader("⚙️ Settings")
    
    # Let user choose routing method
    routing_method = st.selectbox(
        "Intent Detection Method",
        ["Semantic (Fast)", "Keyword (Fastest)", "Disabled"],
        help="Semantic uses embeddings for intelligent routing in any language"
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
# Chains with configurable routing
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

# Choose routing strategy based on user preference
if routing_method == "Disabled":
    # Always use RAG chain
    main_chain = rag_chain
elif routing_method == "Keyword (Fastest)":
    # Simple keyword matching (original method)
    def is_greeting_keyword(text: str) -> bool:
        text = text.lower().strip()
        return any(
            g in text
            for g in ["hi", "hello", "hey", "good morning", "good evening"]
        )
    
    main_chain = RunnableBranch(
        (lambda x: is_greeting_keyword(x), greeting_chain),
        rag_chain,
    )
else:  # Semantic (Fast) - default
    main_chain = RunnableBranch(
        (lambda x: classify_intent_semantic(x), greeting_chain),
        rag_chain,
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
        if routing_method != "Disabled":
            status.write("🧠 Analyzing intent")

        for chunk in main_chain.stream(question):
            if chunk.content:
                streamed_answer += chunk.content
                answer_placeholder.markdown(streamed_answer + "▌")

        status.update(label="✅ Done", state="complete")

    st.session_state.chat_history.append((question, streamed_answer))

# ---------------------------
# Chat history
# ---------------------------
if st.session_state.chat_history:
    st.subheader("📜 Chat History")
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**🧑 Question:** {q}")
        st.markdown(f"**🤖 Answer:** {a}")
        st.divider()
