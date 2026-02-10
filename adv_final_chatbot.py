"""
AI PDF Chatbot with RAG – Working Version
- Fast semantic intent routing
- Streaming responses
- Proper Chroma integration
- Clean chat UI with history
"""

import os
import uuid
import shutil
import hashlib
import numpy as np
import streamlit as st

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

# =============================================================================
# CONFIG
# =============================================================================

class Config:
    CHROMA_DIR = "chroma_db"
    EMBEDDING_MODEL = "all-minilm"
    LLM_MODEL = "qwen2.5:3b-instruct-q4_K_M"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RETRIEVER_K = 4
    LLM_TEMPERATURE = 0.2

# =============================================================================
# STREAMLIT SETUP
# =============================================================================

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI PDF Chatbot — Local RAG with Ollama")
st.caption("Upload PDFs and ask questions using local AI models")

# =============================================================================
# SESSION STATE
# =============================================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = os.path.exists(Config.CHROMA_DIR)

# =============================================================================
# CACHED RESOURCES
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_embeddings():
    """Load embedding model"""
    return OllamaEmbeddings(model=Config.EMBEDDING_MODEL)

@st.cache_resource(show_spinner=False)
def load_llm():
    """Load LLM with streaming enabled"""
    return ChatOllama(
        model=Config.LLM_MODEL,
        temperature=Config.LLM_TEMPERATURE,
        streaming=True,  # Enable streaming for real-time responses
    )

@st.cache_resource(show_spinner=False)
def get_vectorstore():
    """Initialize or load vector store"""
    embeddings = load_embeddings()
    return Chroma(
        persist_directory=Config.CHROMA_DIR,
        embedding_function=embeddings,
    )

@st.cache_resource(show_spinner=False)
def get_intent_embeddings():
    """Pre-compute embeddings for intent classification (fast routing)"""
    embeddings = load_embeddings()
    
    # Greeting examples
    greeting_examples = [
        "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
        "how are you", "what's up", "howdy", "greetings", "hi there",
        "hello there", "hey there", "bonjour", "hola", "namaste",
        "sup", "yo", "good day", "nice to meet you"
    ]
    
    # Document query examples
    document_examples = [
        "what does the document say about",
        "can you explain this section",
        "what is mentioned about",
        "find information on",
        "tell me about the content",
        "summarize the document",
        "what are the key points",
        "explain", "summarize", "what is", "tell me about",
        "find", "search", "look for"
    ]
    
    # Compute embeddings
    greeting_vecs = [embeddings.embed_query(ex) for ex in greeting_examples]
    document_vecs = [embeddings.embed_query(ex) for ex in document_examples]
    
    return {
        "greeting": np.array(greeting_vecs),
        "document": np.array(document_vecs),
        "embeddings": embeddings
    }

# =============================================================================
# PDF PROCESSING
# =============================================================================

def compute_file_hash(data: bytes) -> str:
    """Compute SHA256 hash of file content"""
    return hashlib.sha256(data).hexdigest()

@st.cache_data(show_spinner=False)
def load_and_split_pdf(file_path: str, file_hash: str):
    """Load and split PDF into chunks"""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Add metadata
    for d in docs:
        d.metadata["file_hash"] = file_hash
        d.metadata["source"] = os.path.basename(file_path)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)

# =============================================================================
# INTENT CLASSIFICATION
# =============================================================================

def cosine_similarity(a, b):
    """Compute cosine similarity between vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def classify_intent_semantic(text: str) -> bool:
    """
    Use semantic similarity for FAST intent classification.
    Returns True if greeting, False if document query.
    """
    try:
        intent_data = get_intent_embeddings()
        
        # Embed the user's question
        query_vec = np.array(intent_data["embeddings"].embed_query(text))
        
        # Calculate average similarity to greeting examples
        greeting_sims = [
            cosine_similarity(query_vec, gvec) 
            for gvec in intent_data["greeting"]
        ]
        avg_greeting_sim = np.mean(greeting_sims)
        
        # Calculate average similarity to document examples
        document_sims = [
            cosine_similarity(query_vec, dvec) 
            for dvec in intent_data["document"]
        ]
        avg_document_sim = np.mean(document_sims)
        
        # Return True if more similar to greetings
        return avg_greeting_sim > avg_document_sim
    except Exception:
        # Default to document query on error
        return False

def is_greeting_keyword(text: str) -> bool:
    """Simple keyword-based greeting detection (fastest)"""
    text = text.lower().strip()
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
    return any(g in text for g in greetings)

# =============================================================================
# PROMPTS
# =============================================================================

RAG_PROMPT = PromptTemplate(
    template="""You are an expert AI assistant answering questions from a PDF document.

Use ONLY the provided context to answer.
If the answer is not in the context, say: "I cannot find this information in the uploaded document."

Keep answers concise (2-3 sentences maximum).

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)

GREETING_PROMPT = PromptTemplate.from_template(
    """You are a friendly PDF chatbot assistant. 
Reply briefly and mention you can help answer questions about uploaded PDFs.

User: {question}"""
)

# =============================================================================
# SIDEBAR
# =============================================================================

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
    
    clear_db = st.button("🗑️ Clear Vector Database", use_container_width=True)
    
    st.divider()
    
    # Status indicator
    if st.session_state.vectorstore_ready:
        st.success("✓ Database ready")
    else:
        st.info("⬆ Upload a PDF to begin")

# Handle database clearing
if clear_db:
    if os.path.exists(Config.CHROMA_DIR):
        shutil.rmtree(Config.CHROMA_DIR)
    st.cache_resource.clear()
    st.cache_data.clear()
    st.session_state.vectorstore_ready = False
    st.session_state.chat_history = []
    st.success("✓ Vector database and chat history cleared")
    st.rerun()

# =============================================================================
# INITIALIZE CORE OBJECTS
# =============================================================================

vectorstore = get_vectorstore()
llm = load_llm()

# =============================================================================
# PDF INGESTION
# =============================================================================

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_hash = compute_file_hash(file_bytes)
    
    # Check if already indexed
    try:
        existing = vectorstore.get(where={"file_hash": file_hash})
        if existing and len(existing.get("ids", [])) > 0:
            st.info(f"📄 '{uploaded_file.name}' is already indexed")
            st.session_state.vectorstore_ready = True
        else:
            # Process new PDF
            temp_path = f"temp_{uuid.uuid4()}.pdf"
            
            try:
                with open(temp_path, "wb") as f:
                    f.write(file_bytes)
                
                with st.status(f"📄 Processing '{uploaded_file.name}'...", expanded=True) as status:
                    status.write("📥 Loading & splitting PDF")
                    chunks = load_and_split_pdf(temp_path, file_hash)
                    
                    if chunks:
                        status.write(f"🧠 Storing {len(chunks)} chunks")
                        vectorstore.add_documents(chunks)
                        st.session_state.vectorstore_ready = True
                        status.update(label=f"✅ '{uploaded_file.name}' indexed", state="complete")
                    else:
                        st.error("No content could be extracted from the PDF")
                        
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
    except Exception as e:
        st.error(f"Error processing PDF: {e}")

st.divider()

# =============================================================================
# BUILD CHAINS
# =============================================================================

retriever = vectorstore.as_retriever(search_kwargs={"k": Config.RETRIEVER_K})

# RAG chain for document queries
rag_chain = (
    {
        "context": retriever
        | RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": RunnablePassthrough(),
    }
    | RAG_PROMPT
    | llm
)

# Greeting chain
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
    # Simple keyword matching
    main_chain = RunnableBranch(
        (lambda x: is_greeting_keyword(x), greeting_chain),
        rag_chain,
    )
else:  # Semantic (Fast) - default
    main_chain = RunnableBranch(
        (lambda x: classify_intent_semantic(x), greeting_chain),
        rag_chain,
    )

# =============================================================================
# CHAT UI
# =============================================================================

st.subheader("💬 Ask Questions")

# Question input
question = st.text_input(
    "Ask a question about the document",
    placeholder="What is this document about?",
    key="question_input"
)

# Ask button
if st.button("Ask", type="primary") and question:
    answer_placeholder = st.empty()
    streamed_answer = ""
    
    with st.status("⚙️ Processing...", expanded=True) as status:
        if routing_method != "Disabled":
            # Show which route was chosen
            is_greeting = (
                classify_intent_semantic(question) if routing_method == "Semantic (Fast)"
                else is_greeting_keyword(question)
            )
            route = "Greeting/non-document" if is_greeting else "document query"
            status.write(f"🧠 Understanding Intent: {route}")
        
        # Stream the response
        try:
            for chunk in main_chain.stream(question):
                if hasattr(chunk, 'content') and chunk.content:
                    streamed_answer += chunk.content
                    answer_placeholder.markdown(streamed_answer + "▌")
            
            # Final answer without cursor
            answer_placeholder.markdown(streamed_answer)
            status.update(label="✅ Done", state="complete")
            
            # Add to history
            st.session_state.chat_history.append((question, streamed_answer))
            
        except Exception as e:
            st.error(f"Error generating response: {e}")

# =============================================================================
# CHAT HISTORY
# =============================================================================

if st.session_state.chat_history:
    st.divider()
    st.subheader("📜 Chat History")
    
    for q, a in reversed(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**🧑 Question:** {q}")
            st.markdown(f"**🤖 Answer:** {a}")
            st.divider()