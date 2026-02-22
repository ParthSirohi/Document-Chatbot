"""Simple RAG pipeline - ties everything together"""
from simple_pdf_loader import SimplePDFLoader
from simple_embeddings import SimpleEmbeddings
from simple_vector_store import SimpleVectorStore
from simple_llm import SimpleLLM
from typing import Optional, Tuple, List, Dict

class SimpleRAG:
    def __init__(self, use_groq: bool = True, llm_api_key: Optional[str] = None):
        self.loader = SimplePDFLoader()
        self.embeddings = SimpleEmbeddings()
        self.vector_store = SimpleVectorStore()
        self.llm = SimpleLLM(use_groq=use_groq, api_key=llm_api_key)
    
    def initialize(self):
        """Load PDFs and create embeddings index if not exists"""
        
        # Try to load existing index
        if self.vector_store.load_index():
            print("\n[OK] Using existing index\n")
            return
        
        print("\n[*] Creating new index from PDFs...\n")
        
        # Load PDFs
        documents = self.loader.load_pdfs()
        if not documents:
            print("[!] No PDFs found!")
            return
        
        # Create embeddings
        texts = [doc['content'] for doc in documents]
        embeddings = self.embeddings.create_embeddings(texts)
        
        # Create and save index
        self.vector_store.create_index(embeddings, documents)
        print("\n[OK] Index ready!\n")
    
    @staticmethod
    def _format_chat_history(chat_history: Optional[List[Dict]], max_turns: int = 4) -> str:
        if not chat_history:
            return ""
        recent = chat_history[-max_turns:]
        lines = []
        for msg in recent:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def answer(
        self,
        query: str,
        num_results: int = 5,
        chat_history: Optional[List[Dict]] = None
    ) -> Tuple[str, list]:
        """
        Answer a query using RAG
        
        Returns:
            (answer, sources) where sources are the retrieved documents
        """
        
        print(f"\n[*] Processing query: {query}")
        print(f"[*] Searching {len(self.vector_store.documents)} documents...")
        
        # Embed query
        query_embedding = self.embeddings.embed_query(query)
        
        # Search for relevant documents
        search_results = self.vector_store.search(query_embedding, query_text=query, k=num_results)
        
        if not search_results:
            return "No relevant documents found.", []
        
        # Combine context from search results
        context_parts = []
        for i, doc in enumerate(search_results, 1):
            source = doc.get('source', 'unknown')
            page = doc.get('page', '?')
            chunk = doc.get('chunk')
            chunk_text = f", chunk {chunk}" if chunk else ""
            context_parts.append(
                f"[Doc {i} | {source} | page {page}{chunk_text}]\n{doc.get('content', '')}"
            )
        context = "\n\n".join(context_parts)
        history_text = self._format_chat_history(chat_history)
        
        # Generate answer using LLM
        print(f"[*] Generating answer from LLM...")
        answer = self.llm.answer_query(query, context, chat_history=history_text)
        
        print(f"[OK] Done!\n")
        
        return answer, search_results


def main():
    """Simple test"""
    
    # Initialize RAG
    rag = SimpleRAG(use_groq=False)  # Using Ollama (local)
    rag.initialize()
    
    # Answer a query
    query = "What are the capital requirements for banks?"
    answer, sources = rag.answer(query)
    
    print("=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)
    print(f"\nANSWER:\n{answer}\n")
    print("=" * 70)
    print(f"SOURCES:")
    for src in sources:
        print(f"  - {src['source']}, page {src['page']}")


if __name__ == "__main__":
    main()
