# ============================================================
# ast_validator.py  —  DOMAIN-AGNOSTIC STATIC VALIDATOR
# ============================================================
import ast

FORBIDDEN_CALLS  = {"eval", "exec", "open", "compile", "subprocess", "__import__"}
ALLOWED_IMPORTS  = {"torch"}
REQUIRED_FUNC    = "_get_rewards"

def validate_ast(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if REQUIRED_FUNC not in func_names:
        return False, f"`{REQUIRED_FUNC}` function not found"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in getattr(node, "names", []):
                mod = alias.name.split(".")[0]
                if mod not in ALLOWED_IMPORTS:
                    return False, f"Illegal import: {alias.name}"
        if isinstance(node, ast.Call):
            func = node.func
            name = (func.id if isinstance(func, ast.Name) else
                    func.attr if isinstance(func, ast.Attribute) else None)
            if name in FORBIDDEN_CALLS:
                return False, f"Forbidden call: {name}()"

    return True, "OK"