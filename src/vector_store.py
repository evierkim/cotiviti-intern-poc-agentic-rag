import pickle
import os
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_EMBEDDING_MODEL: SentenceTransformer | None = None
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Reuse one SentenceTransformer instance across VectorStore rebuilds."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer(model_name)
    return _EMBEDDING_MODEL


class VectorStore:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model = get_embedding_model(model_name)
        self.index = None
        self.documents = []
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension

    def clear(self):
        """Drop indexed chunks without reloading the embedding model."""
        self.index = None
        self.documents = []

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Add document chunks to vector store."""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        # Initialize FAISS index if needed
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add to FAISS index
        self.index.add(np.array(embeddings).astype(np.float32))
        self.documents.extend(chunks)
    
    def search(self, query: str, k: int = 2) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        if self.index is None or self.index.ntotal == 0:
            return []

        query_embedding = self.model.encode([query], show_progress_bar=False)
        distances, indices = self.index.search(
            np.array(query_embedding).astype(np.float32),
            min(k, len(self.documents))
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'score': float(distances[0][i])
                })
        
        return results
    
    def save(self, path: str):
        """Save vector store to disk."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
    
    def load(self, path: str):
        """Load vector store from disk."""
        self.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/documents.pkl", 'rb') as f:
            self.documents = pickle.load(f)