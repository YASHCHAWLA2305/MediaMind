import sys
import os
import ast
import shlex

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "Agents"))

from llm_wrapper import get_llm
from Agents.calculation_agent import execute_python_code, clean_code, match_function
from Agents.text_agent import query_resume  # Resume RAG agent

from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Centralized memory
from Memory import memory, faiss_index, debug_memory_state


# Initialize LLM
llm = get_llm()

# Calculator Tool
def safe_calculation_wrapper(query: str) -> str:
    """
    Generates Python code, cleans, matches function, and executes safely.
    """
    try:
        template = PromptTemplate(input_variables=["task"], template="{task}")
        chain = LLMChain(llm=llm, prompt=template, memory=memory, verbose=True)
        raw_code = chain.invoke({"task": query})["text"]

        safe_code = clean_code(raw_code)

        try:
            tree = ast.parse(safe_code)
            func_names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
        except Exception:
            func_names = []

        func_name = match_function(func_names, query)

        result = execute_python_code(safe_code, func_name=func_name, func_args=())
        if result.get("error"):
            return f"Error: {result['error']}"
        return str(result["output"])
    except Exception as e:
        return f"Calculation failed: {e}"


# Define Tools / Agents
tools = [
    Tool(
        name="Calculator",
        func=safe_calculation_wrapper,
        description="Executes safe Python calculations. Use for math/code tasks."
    ),
    Tool(
        name="ResumeRAG",
        func=query_resume,
        description="Answer questions about your resume using RAG embeddings."
    ),
    Tool(
        name="ChatBot",
        func=lambda q: llm.invoke(q),
        description="General-purpose chatbot for open-ended conversation."
    ),
]

# Unified Conversational Agent
agent = initialize_agent(
    tools,
    llm,
    agent="conversational-react-description",
    memory=memory,
    verbose=True
)

# Tool Logging Wrapper

def run_with_logging(user_input: str) -> str:
    """
    Runs the agent and logs which tool was chosen.
    """
    response = agent.run(user_input)
    tool_used = getattr(agent, "last_tool_used", "Unknown")
    print(f"[DEBUG] Tool used: {tool_used}")

    return response

# Interaction Loop
if __name__ == "__main__":
    print("Agent ready. Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Run agent with tool logging
        response = run_with_logging(user_input)
        print("Agent:", response)

        # Save interaction to FAISS (long-term memory)
        faiss_index.add_texts([f"User: {user_input}\nAgent: {response}"])
        faiss_index.save_local("faiss_index")

        # Debug: show memory state
        debug_memory_state(query=user_input)
