"""Simple PDF loader - load PDFs and split pages into robust text chunks."""
import os
import re
import fitz  # PyMuPDF
from typing import List, Dict

class SimplePDFLoader:
    def __init__(
        self,
        pdf_folder: str = "pdf_data",
        chunk_size: int = 1400,
        chunk_overlap: int = 220,
        min_chunk_chars: int = 220,
    ):
        self.pdf_folder = pdf_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars
        self.documents = []

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize whitespace and remove noisy line breaks."""
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks while respecting paragraph boundaries.
        """
        clean_text = self._normalize_text(text)
        if not clean_text:
            return []

        paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]
        chunks: List[str] = []
        current = ""

        for para in paragraphs:
            candidate = para if not current else f"{current}\n\n{para}"
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current and len(current) >= self.min_chunk_chars:
                chunks.append(current)
            elif len(para) > self.chunk_size:
                # Hard split large paragraphs.
                start = 0
                step = max(self.chunk_size - self.chunk_overlap, 120)
                while start < len(para):
                    piece = para[start:start + self.chunk_size].strip()
                    if piece and len(piece) >= self.min_chunk_chars:
                        chunks.append(piece)
                    start += step
                current = ""
                continue

            if chunks and self.chunk_overlap > 0:
                overlap_text = chunks[-1][-self.chunk_overlap:].strip()
                current = f"{overlap_text}\n\n{para}".strip()
            else:
                current = para

        if current and len(current) >= self.min_chunk_chars:
            chunks.append(current)

        return chunks
    
    def load_pdfs(self) -> List[Dict]:
        """Load all PDFs from pdf_data folder"""
        documents = []
        
        if not os.path.exists(self.pdf_folder):
            print(f"[!] Folder '{self.pdf_folder}' not found")
            return documents
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.pdf_folder):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    print(f"[*] Loading: {pdf_path}")
                    
                    try:
                        doc = fitz.open(pdf_path)
                        
                        for page_num in range(len(doc)):
                            page = doc[page_num]
                            text = page.get_text()

                            page_chunks = self._split_text(text)
                            for chunk_idx, chunk in enumerate(page_chunks, 1):
                                documents.append({
                                    'content': chunk,
                                    'source': os.path.relpath(pdf_path),
                                    'page': page_num + 1,
                                    'chunk': chunk_idx,
                                    'chunk_count': len(page_chunks),
                                })
                    except Exception as e:
                        print(f"[!] Error loading {pdf_path}: {e}")
        
        self.documents = documents
        print(f"\n[OK] Loaded {len(documents)} pages from PDFs\n")
        return documents


if __name__ == "__main__":
    loader = SimplePDFLoader()
    docs = loader.load_pdfs()
    
    for i, doc in enumerate(docs[:3]):
        print(f"\n--- Document {i+1} ---")
        print(f"Source: {doc['source']}, Page: {doc['page']}")
        print(f"Text preview: {doc['content'][:200]}...")
