# llm_wrapper.py
from langchain_community.llms import Ollama

def get_llm():
    """
    Return an Ollama LLM instance using Gemma 2B (local).
    """
    llm = Ollama(model="gemma:2b")  
    return llm
