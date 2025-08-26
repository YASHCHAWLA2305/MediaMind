# Agents/calculation_agent.py
import ast
import difflib
import io
import contextlib

def clean_code(code: str) -> str:
    """
    Cleans the LLM-generated code:
    - Keeps only function definitions
    """
    try:
        tree = ast.parse(code)
        safe_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        cleaned_code = ""
        for node in safe_nodes:
            cleaned_code += ast.unparse(node) + "\n\n"
        return cleaned_code.strip()
    except Exception:
        return ""

def match_function(func_names: list, user_input: str) -> str:
    """
    Match user_input to the closest function name from func_names.
    Returns the best match or first function if no good match.
    """
    if not func_names:
        return None
    if user_input:
        matches = difflib.get_close_matches(user_input.lower(), func_names, n=1, cutoff=0.3)
        if matches:
            return matches[0]
    return func_names[0]

def execute_python_code(code: str, func_args: tuple = (), func_name: str = None):
    """
    Execute cleaned Python code safely.
    - If func_name is provided, runs that function.
    - If no func_name, but a `main` function exists, calls main(*func_args).
    - Otherwise, returns top-level variables.
    - Captures stdout from print().
    """
    result = {"output": None, "error": None}
    local_env = {}
    stdout_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, {}, local_env)

            # Explicit function call
            if func_name:
                func = local_env.get(func_name)
                if not func:
                    return {"output": None, "error": f"Function '{func_name}' not found"}
                result["output"] = func(*func_args)

            # Default to main() if present
            elif "main" in local_env and callable(local_env["main"]):
                result["output"] = local_env["main"](*func_args)

        # If nothing was returned, fallback to stdout or variables
        if result["output"] is None:
            stdout_val = stdout_buffer.getvalue().strip()
            if stdout_val:
                result["output"] = stdout_val
            else:
                result["output"] = {k: v for k, v in local_env.items() if not k.startswith("__")}

    except Exception as e:
        result["error"] = str(e)
    finally:
        stdout_buffer.close()

    return result
