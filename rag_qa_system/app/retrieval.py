import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class DocumentRetriever:
    def __init__(self, knowledge_base_dir, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.knowledge_base_dir = knowledge_base_dir
        self.index = None
        self.documents = []

        self._load_documents()
        self._build_index()

    def _load_documents(self):
        """Load documents from the knowledge base directory."""
        for filename in os.listdir(self.knowledge_base_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(self.knowledge_base_dir, filename), "r") as file:
                    self.documents.append(file.read())

    def _build_index(self):
        """Build a FAISS index for the documents."""
        embeddings = self.model.encode(self.documents)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def retrieve(self, query, top_k=3):
        """Retrieve the top-k most relevant documents for a query."""
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices[0]]