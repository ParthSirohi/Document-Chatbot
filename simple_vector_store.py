"""Simple vector store - just store and retrieve using FAISS"""
import numpy as np
import faiss
import os
import json
import re
from typing import List, Dict

class SimpleVectorStore:
    def __init__(self, index_path: str = "storage"):
        self.index_path = index_path
        self.index = None
        self.documents = []
        self.index_version = "v2_chunked_hybrid"
        os.makedirs(index_path, exist_ok=True)

    @staticmethod
    def _tokenize(text: str) -> set:
        return set(re.findall(r"[a-z0-9]{3,}", text.lower()))

    def _keyword_score(self, query_tokens: set, doc_text: str) -> float:
        """Simple lexical overlap score in [0, 1]."""
        if not query_tokens:
            return 0.0
        doc_tokens = self._tokenize(doc_text)
        overlap = len(query_tokens.intersection(doc_tokens))
        return overlap / max(len(query_tokens), 1)
    
    def create_index(self, embeddings: np.ndarray, documents: List[Dict]):
        """Create FAISS index from embeddings"""
        print(f"[*] Creating FAISS index...")
        
        # With normalized vectors, inner-product behaves like cosine similarity.
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
        self.documents = documents
        
        print(f"[OK] Created index with {len(documents)} documents")
        self._save_index()
    
    def _save_index(self):
        """Save index and documents to disk"""
        # Save index
        index_file = os.path.join(self.index_path, "faiss_index.bin")
        faiss.write_index(self.index, index_file)
        
        # Save documents metadata
        docs_file = os.path.join(self.index_path, "documents.json")
        with open(docs_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

        meta_file = os.path.join(self.index_path, "index_meta.json")
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump({"index_version": self.index_version}, f, ensure_ascii=False, indent=2)
        
        print(f"[OK] Saved index to {index_file}")
        print(f"[OK] Saved documents to {docs_file}")
    
    def load_index(self):
        """Load index and documents from disk"""
        index_file = os.path.join(self.index_path, "faiss_index.bin")
        docs_file = os.path.join(self.index_path, "documents.json")
        meta_file = os.path.join(self.index_path, "index_meta.json")
        
        if not os.path.exists(index_file) or not os.path.exists(docs_file) or not os.path.exists(meta_file):
            return False
        
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            if meta.get("index_version") != self.index_version:
                print("[*] Existing index version is outdated. Rebuilding.")
                return False

            self.index = faiss.read_index(index_file)
            with open(docs_file, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            print(f"[OK] Loaded index with {len(self.documents)} documents")
            return True
        except Exception as e:
            print(f"[!] Error loading index: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, query_text: str, k: int = 5) -> List[Dict]:
        """Search using semantic retrieval + lightweight keyword reranking."""
        if self.index is None:
            return []
        
        fetch_k = min(max(k * 4, 12), len(self.documents))
        similarities, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            fetch_k
        )
        
        query_tokens = self._tokenize(query_text)
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):
                doc = self.documents[idx].copy()
                semantic_score = float(sim)
                lexical_score = self._keyword_score(query_tokens, doc.get('content', ''))
                hybrid_score = (0.78 * semantic_score) + (0.22 * lexical_score)
                doc['semantic_score'] = semantic_score
                doc['keyword_score'] = lexical_score
                doc['score'] = hybrid_score
                results.append(doc)

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]


if __name__ == "__main__":
    # Test
    embeddings = np.random.rand(10, 384).astype('float32')
    docs = [
        {'content': f'Test doc {i}', 'source': f'doc_{i}.pdf', 'page': 1}
        for i in range(10)
    ]
    
    store = SimpleVectorStore()
    store.create_index(embeddings, docs)
    
    # Test search
    query_emb = np.random.rand(384).astype('float32')
    results = store.search(query_emb, k=3)
    print(f"\nFound {len(results)} results")
    for r in results:
        print(f"  - {r['source']}, page {r['page']}, score: {r['score']:.2f}")
