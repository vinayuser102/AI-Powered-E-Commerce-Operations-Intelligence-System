import os
import chromadb

def run_ingestion():
    print("📦 Starting SmartOps Knowledge Base Ingestion...")
    
    # 1. Read the raw document text
    with open("knowledge_base.txt", "r") as f:
        raw_text = f.read()
    
    # 2. Basic Chunking: Split by our explicit chunk tags
    chunks = [chunk.strip() for chunk in raw_text.split("\n\n") if chunk.strip()]
    ids = [f"smartops_doc_{i}" for i in range(len(chunks))]
    
    # 3. Initialize a local, persistent ChromaDB on the drive
    # This creates a folder named 'chroma_storage' inside your module directory
    chroma_client = chromadb.PersistentClient(path="./chroma_storage")
    
    # 4. Create a collection (uses Chroma's default text embedding model)
    collection = chroma_client.get_or_create_collection(name="ops_intelligence")
    
    # 5. Upsert raw text chunks directly (Chroma automatically handles embedding math)
    collection.upsert(
        documents=chunks,
        ids=ids
    )
    
    print(f"✅ Success! Indexed {len(chunks)} operational chunks into ChromaDB.")

if __name__ == "__main__":
    run_ingestion()




