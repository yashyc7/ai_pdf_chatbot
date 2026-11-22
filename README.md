# AI PDF Chatbot (Local RAG with Ollama)

Upload a PDF and ask questions about it. Everything runs **locally on your machine**: the language model and embeddings come from [Ollama](https://ollama.com/), the vectors are stored in [ChromaDB](https://www.trychroma.com/), and the UI is built with [Streamlit](https://streamlit.io/). No API key and no data leaves your computer.

## How it works (RAG)

1. **Load and split**: the PDF is read with `PyPDFLoader` and cut into chunks of 1000 characters with 200 overlap (`RecursiveCharacterTextSplitter`).
2. **Embed and store**: each chunk is converted to a vector with the `all-minilm` model and saved in ChromaDB (`chroma_db/` folder).
3. **Retrieve**: for every question, the 4 most similar chunks are fetched.
4. **Answer**: the chunks and the question go into a prompt for `qwen2.5:3b-instruct-q4_K_M`, which is told to answer **only from the document**. The answer streams to the screen token by token.
5. **Greeting routing**: messages like "hi" or "hello" are sent to a small greeting chain instead of searching the PDF.

## Versions in this repo

There are three Streamlit apps, each one an improvement over the last:

| File | What it adds |
|------|--------------|
| `main.py` | Basic version. PDF upload, RAG chain, streaming answer, keyword-based greeting detection, clear database button |
| `advanced_chatbot.py` | Selectable intent detection: **Semantic** (embedding similarity with example phrases), **Keyword**, or **Disabled** |
| `adv_final_chatbot.py` | **Recommended.** Everything above, plus: config class, SHA-256 file hash so the same PDF is not indexed twice, "database ready" status, error handling, chat history, shorter answers (2-3 sentences), clears chat history with the database |

## Setup

### Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running

### Steps

1. Pull the two models:
   ```bash
   ollama pull all-minilm
   ollama pull qwen2.5:3b-instruct-q4_K_M
   ```
2. Clone the repo and install packages:
   ```bash
   git clone https://github.com/yashyc7/ai_pdf_chatbot.git
   cd ai_pdf_chatbot
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install streamlit numpy pypdf chromadb langchain langchain-community langchain-text-splitters langchain-ollama
   ```
3. Run the recommended app:
   ```bash
   streamlit run adv_final_chatbot.py
   ```
4. Open the link shown in the terminal (usually `http://localhost:8501`), upload a PDF from the sidebar, and start asking questions.

To try the older versions, run `streamlit run main.py` or `streamlit run advanced_chatbot.py`.

## Configuration

In `adv_final_chatbot.py`, all settings are in the `Config` class:

| Setting | Default |
|---------|---------|
| `EMBEDDING_MODEL` | `all-minilm` |
| `LLM_MODEL` | `qwen2.5:3b-instruct-q4_K_M` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` |
| `RETRIEVER_K` | `4` |
| `LLM_TEMPERATURE` | `0.2` |

You can use any other model from Ollama by changing the model names.

## Tech stack

Python, Streamlit, LangChain, Ollama (`qwen2.5` and `all-minilm`), ChromaDB, PyPDF, NumPy

## Notes

- All uploaded PDFs go into the same vector database, so answers can come from any of them. Use **Clear Vector Database** to start fresh.
- The app only works with PDFs that have selectable text. Scanned PDFs need OCR first.
- A small 3B model is used so it runs on a normal laptop. Answers will be better with a bigger model.
