import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

class RAGIngestor:
    def __init__(self, chroma_dir: str = None, collection_name: str = "smartops_policies"):
        # Automatically resolve path to rag_qa_engine/chroma_storage if not specified
        if chroma_dir is None:
            base_dir = Path(__file__).resolve().parent
            chroma_dir = str(base_dir / "chroma_storage")
            
        self.chroma_dir = chroma_dir
        self.client = chromadb.PersistentClient(path=self.chroma_dir)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            embedding_function=self.embedding_fn
        )

    def ingest_text_file(self, file_path: str):
        """Reads operational policy text file, chunks sections, and stores vector embeddings in ChromaDB."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at path: {file_path}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Split text by double line breaks (paragraphs/sections)
        documents = [doc.strip() for doc in text.split("\n\n") if doc.strip()]
        ids = [f"policy_{i+1}" for i in range(len(documents))]

        self.collection.add(
            documents=documents,
            ids=ids
        )
        print(f"✅ Ingested {len(documents)} policy blocks into ChromaDB at '{self.chroma_dir}'.")

# Self-execution block when running directly from terminal
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    kb_path = BASE_DIR / "knowledge_base.txt"
    
    ingestor = RAGIngestor()
    ingestor.ingest_text_file(str(kb_path))