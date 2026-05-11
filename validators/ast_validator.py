import ast
import re

ALLOWED_IMPORTS = {"torch"}
FORBIDDEN_CALLS = {"exec", "eval", "open", "__import__", "compile", "input"}
FORBIDDEN_MODULE_NAMES = {"os", "sys", "subprocess", "pathlib", "shutil"}

# Accept either IsaacLab style (_get_rewards) or standalone style (compute_reward)
VALID_FUNCTION_NAMES = {"_get_rewards", "compute_reward"}


def static_validate(code: str) -> tuple[bool, str]:
    if not re.search(r"\bdef\s+\w+\s*\(", code):
        return False, "No function definition found — code may be truncated"

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    found_fn = False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in VALID_FUNCTION_NAMES:
            found_fn = True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    return False, f"Forbidden import: {alias.name}"
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if not module.startswith("torch"):
                return False, f"Forbidden ImportFrom: {module}"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                return False, f"Forbidden call: {node.func.id}"
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_MODULE_NAMES:
            return False, f"Forbidden module reference: {node.id}"

    if not found_fn:
        return (
            False,
            f"Neither '_get_rewards' nor 'compute_reward' function found. "
            f"LLM used a different function name.",
        )

    return True, "OK"