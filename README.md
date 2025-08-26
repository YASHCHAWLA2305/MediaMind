# MediaMind
This project is an AI-powered dynamic execution agent that takes natural language instructions from the user, generates the required Python code using an LLM, safely cleans the code, and executes it dynamically.Additionally, The agent can classify the type of task before execution, identifying whether the instruction is:

A RAG (Retrieval-Augmented Generation) action

A general query

A mathematical/computational task

This enables intelligent routing and safe execution, making the system flexible and reliable for a wide variety of tasks.

Key Features
•	Natural Language to Code → Users can simply describe tasks (e.g., “open an Excel file and show me the first 5 rows”) without worrying about function names.
•	Dynamic Function Matching → The agent automatically detects and runs the right function from the generated code, even if the user doesn’t know the exact function name.
•	Excel-Friendly → Supports opening and processing Excel files (via pandas + openpyxl) with just a file path input.
•	Safe Execution → Code is sandboxed: only function definitions are retained, and dangerous operations (like exec, eval, system calls) are stripped.
•	Extensible → Add your own agents, utility functions, or domain-specific modules for different workflows (e.g., media audit, data validation).

🔧 Tech Stack
•	Python 3.10+
•	LangChain (LLM orchestration)
•	OpenAI / other LLMs for code generation
•	pandas for data handling
•	openpyxl for Excel processing

🚀 Use Cases
•	Automating Excel-based workflows (opening, validating, comparing datasets)
•	Building an AI assistant for data analysts that bridges natural language and code execution
•	Dynamic media audit & reporting agents
