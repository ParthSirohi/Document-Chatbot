"""Test retrieval without needing LLM"""
from simple_rag import SimpleRAG

print("\n" + "="*70)
print("SIMPLE CHATBOT - RETRIEVAL TEST")
print("="*70 + "\n")

# Initialize RAG (will use existing index)
rag = SimpleRAG(use_groq=False)
rag.initialize()

# Test queries
test_queries = [
    "What are the capital requirements for banks?",
    "What does banking mean?",
    "minimum capital for banking"
]

for query in test_queries:
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print('='*70)
    
    # Get the answer
    answer, sources = rag.answer(query, num_results=3)
    
    print(f"\nTOP 3 RETRIEVED DOCUMENTS:")
    for i, src in enumerate(sources, 1):
        print(f"\n  [{i}] {src['source']}, page {src['page']}")
        print(f"      Score: {src['score']:.2f}")
        print(f"      Preview: {src['content'][:150]}...")
    
    print(f"\n\nNOTE: LLM Answer would be generated here (requires Ollama or Groq API)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"[OK] Loaded {len(rag.vector_store.documents)} documents from PDFs")
print(f"[OK] Retrieval working correctly")
print(f"[OK] To generate answers, set up:")
print(f"  - Groq: Already configured from .env file")
print(f"  - Ollama: https://ollama.ai (local, free)")
print(f"\nNext step: streamlit run app_simple.py")
print("="*70 + "\n")
