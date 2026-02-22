# Document Chatbot 📄💬

A RAG (Retrieval-Augmented Generation) chatbot built with Streamlit that allows you to upload PDF documents and ask questions about them.

## Features

- 📤 Upload PDF files through the web interface
- 🤖 Powered by Groq API (llama-3.3-70b-versatile) or local Ollama
- 🔍 Semantic search using BAAI/bge-small-en-v1.5 embeddings
- 💾 Persistent vector index with FAISS
- 💬 ChatGPT-like interface
- 📚 Shows source citations for answers

## Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd my_doc_bot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at: https://console.groq.com/

### 4. Run the app
```bash
streamlit run app_simple.py
```

## Deployment on Streamlit Cloud

1. Push this repository to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and select this repository
4. In **Advanced settings**, add your secrets:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Deploy!

## Usage

1. Upload PDF files using the sidebar uploader
2. Click "Save & Reindex" to process the documents
3. Ask questions in the chat interface
4. View source citations for each answer

## Tech Stack

- **Streamlit** - Web interface
- **PyMuPDF** - PDF processing
- **Sentence Transformers** - Text embeddings
- **FAISS** - Vector similarity search
- **Groq API** - LLM inference

## License

MIT
