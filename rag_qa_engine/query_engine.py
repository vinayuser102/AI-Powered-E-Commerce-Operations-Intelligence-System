import os
import chromadb
from groq import Groq

# 1. Connect to the local ChromaDB index
chroma_client = chromadb.PersistentClient(path="./chroma_storage")
collection = chroma_client.get_collection(name="ops_intelligence")

# 2. Initialize the Groq client
groq_client = Groq()

def ask_smartops(user_question: str):
    print(f"\n🔍 User Question: '{user_question}'")
    print("📡 Querying ChromaDB Vector Index (Fetching Top 2 Matches)...")
    
    # PHASE 1: RETRIEVAL (Pulling multiple chunks to prevent missing data)
    search_results = collection.query(
        query_texts=[user_question],
        n_results=3
    )
    
    # Combine all found documents into a single text block
    retrieved_chunks = search_results["documents"][0]
    retrieved_context = "\n\n".join(retrieved_chunks)
    
    print("\n🎯 Context Chunks Found:")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"--- Chunk {i} ---\n{chunk[:150]}...")
    
    # PHASE 2: DEFENSIVE PROMPT CONSTRUCTION
    system_prompt = (
        "You are the SmartOps AI Copilot. Answer the user's question using ONLY the provided context. "
        "If the answer cannot be found in the context, say 'Information not found in operational logs.'"
    )
    
    user_prompt = f"Context:\n{retrieved_context}\n\nQuestion: {user_question}"
    
    # PHASE 3: GENERATION (Calling active Groq Llama 3.1 model)
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="llama-3.1-8b-instant",
        temperature=0.0
    )
    
    print("\n🤖 SmartOps Response:")
    print(chat_completion.choices[0].message.content)

if __name__ == "__main__":
    ask_smartops('What is the target response time for Tier-1 customer support tickets?')