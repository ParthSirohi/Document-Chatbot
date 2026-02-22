"""Simple Streamlit Chatbot UI - like ChatGPT"""
import streamlit as st
from simple_rag import SimpleRAG
import os
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page config
st.set_page_config(
    page_title="Document Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Get Groq API key from .env file
    default_api_key = os.getenv("GROQ_API_KEY", "")
    
    # Use Groq by default if API key is set
    use_groq = st.checkbox("Use Groq API", value=True if default_api_key else False, 
                          help="Enable to use Groq (auto-loaded from .env)\nDisable to use local Ollama")
    
    if use_groq:
        # Pre-fill with API key from .env
        if default_api_key:
            api_key = st.text_input("Groq API Key", type="password", 
                                   value=default_api_key,
                                   disabled=True)
            st.caption("✅ API Key loaded from .env file")
        else:
            api_key = st.text_input("Groq API Key", type="password", 
                                   value="",
                                   placeholder="Paste your Groq API key here")
    else:
        api_key = None
        st.info("💡 Using Ollama locally. Make sure it's running on localhost:11434")
    
    num_results = st.slider("Number of documents to retrieve", 1, 12, 5)
    
    st.markdown("---")
    st.subheader("📤 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to add to your knowledge base"
    )
    
    if uploaded_files:
        if st.button("💾 Save & Reindex"):
            pdf_folder = "pdf_data"
            os.makedirs(pdf_folder, exist_ok=True)
            
            saved_files = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(pdf_folder, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_files.append(uploaded_file.name)
            
            st.success(f"✅ Saved {len(saved_files)} file(s)")
            
            # Force reindex by deleting existing index
            if 'rag' in st.session_state:
                del st.session_state['rag']
            
            # Delete the index files to force recreation
            storage_path = "storage"
            if os.path.exists(storage_path):
                shutil.rmtree(storage_path)
            
            st.info("🔄 Reindexing documents...")
            st.rerun()
    
    if st.button("🔄 Reload Index"):
        if 'rag' in st.session_state:
            del st.session_state['rag']
        st.rerun()

# Main content
st.title("📄 Document Chatbot")
st.markdown("---")

# Initialize RAG if not already done
if 'rag' not in st.session_state:
    with st.spinner("⏳ Initializing chatbot..."):
        st.session_state.rag = SimpleRAG(use_groq=use_groq, llm_api_key=api_key)
        st.session_state.rag.initialize()

# Initialize chat history if not exists
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Sources"):
                for src in message["sources"]:
                    st.caption(f"📄 {src['source']} (page {src['page']})")

# User input
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("💭 Thinking..."):
            answer, sources = st.session_state.rag.answer(
                user_input,
                num_results=num_results,
                chat_history=st.session_state.messages
            )
        
        st.markdown(answer)
        
        # Show sources in expander
        if sources:
            with st.expander("📚 Sources"):
                for src in sources:
                    st.caption(f"📄 {src['source']} (page {src['page']})")
    
    # Add assistant response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🆕 New Chat"):
        st.session_state.messages = []
        st.rerun()
with col2:
    if st.button("🗑️ Clear History"):
        st.session_state.messages = []
        st.rerun()
with col3:
    st.caption(f"📊 {len(st.session_state.rag.vector_store.documents)} documents indexed")
