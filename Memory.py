# Memory.py

import os
from langchain.memory import ConversationBufferMemory, CombinedMemory, VectorStoreRetrieverMemory
from langchain_community.memory.kg import ConversationKGMemory
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM


llm = OllamaLLM(model="gemma:2b")




# Paths

DATA_DIR = "data"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")


os.makedirs(DATA_DIR, exist_ok=True)


# Embeddings

embedding = OllamaEmbeddings(model="mxbai-embed-large")  # Consistent embedding for all memory


if os.path.exists(FAISS_INDEX_PATH):
    faiss_index = FAISS.load_local(
        FAISS_INDEX_PATH,
        embedding,
        allow_dangerous_deserialization=True  
    )
else:
  
    faiss_index = FAISS.from_texts(["Agent memory initialized"], embedding,allow_dangerous_deserialization=True)
    faiss_index.save_local(FAISS_INDEX_PATH)

# Retriever
retriever = faiss_index.as_retriever(search_kwargs={"k": 3})
vector_memory = VectorStoreRetrieverMemory(retriever=retriever)

# Short-term conversation buffer
buffer_memory = ConversationBufferMemory(memory_key="buffer_history")

# Long-term vector memory
vector_memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key="vector_history")

# Knowledge Graph memory
kg_memory = ConversationKGMemory(llm=llm, memory_key="kg_history")  # If supported



# Combined hybrid memory

memory = CombinedMemory(memories=[buffer_memory, vector_memory, kg_memory])

# Debug helper

def debug_memory_state(query: str = "memory"):
    """
    Inspect current FAISS vector memory contents.
    """
    docs = faiss_index.similarity_search(query, k=5)
    for i, doc in enumerate(docs, 1):
        print(f"[{i}] {doc.page_content}")
