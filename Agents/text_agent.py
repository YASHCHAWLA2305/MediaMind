# text_agent.py
import os
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from llm_wrapper import get_llm


# Paths

DATA_DIR = "data"
RESUME_FILENAME = "Resume__XXXX"
RESUME_PATH = os.path.join(DATA_DIR, RESUME_FILENAME)
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "resume_faiss_index")

# Load resume text

def load_docx(path: str) -> str:
    """Load a DOCX file and return all text as a single string."""
    doc = Document(path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


# Embeddings & FAISS setup

embedding = OllamaEmbeddings(model="mxbai-embed-large")  

if os.path.exists(FAISS_INDEX_PATH):
    # Load existing FAISS index
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embedding,allow_dangerous_deserialization=True)
else:
    # Load resume
    resume_text = load_docx("XXXX")

    # Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(resume_text)

    # Build FAISS index and save
    vectorstore = FAISS.from_texts(chunks, embedding,allow_dangerous_deserialization=True)
    vectorstore.save_local(FAISS_INDEX_PATH)

# Retriever for RAG
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# LLM & RAG Chain

llm = get_llm()  

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)


# Query function (importable)

def query_resume(query: str) -> str:
    """
    Query the resume using RAG and return answer as string.
    """
    return qa_chain.run(query)


# Debug function

def debug_resume_retrieval(query: str):
    """Print the top 5 retrieved chunks for debugging."""
    docs = retriever.get_relevant_documents(query)
    for i, doc in enumerate(docs, 1):
        print(f"[{i}] {doc.page_content}\n")
