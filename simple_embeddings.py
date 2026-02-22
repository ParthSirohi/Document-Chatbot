"""Simple embeddings using sentence-transformers."""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class SimpleEmbeddings:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        BGE-small is still lightweight but materially better for retrieval quality.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        print(f"[OK] Loaded embedding model: {model_name}")
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Convert texts to embeddings"""
        print(f"[*] Creating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        print(f"[OK] Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query"""
        return self.model.encode([query], normalize_embeddings=True)[0]


if __name__ == "__main__":
    emb = SimpleEmbeddings()
    
    # Test
    test_texts = [
        "The capital requirement for banks is 10%",
        "Banks need minimum equity"
    ]
    embeddings = emb.create_embeddings(test_texts)
    print(f"Embeddings shape: {embeddings.shape}")
