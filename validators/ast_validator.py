# ============================================================
# ast_validator.py  —  DOMAIN-AGNOSTIC STATIC VALIDATOR
# ============================================================
import ast

FORBIDDEN_CALLS = {"eval", "exec", "open", "compile", "subprocess", "__import__"}
ALLOWED_IMPORTS = {"torch"}
REQUIRED_FUNC   = {"_get_rewards"}


def validate_ast(code: str, required_func: set = REQUIRED_FUNC) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if not required_func.intersection(func_names):                  # ← THE FIX
        return False, f"`{required_func}` function not found — got: {func_names}"

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

# # ============================================================
# # ast_validator.py  —  DOMAIN-AGNOSTIC STATIC VALIDATOR
# # ============================================================
# import ast

# FORBIDDEN_CALLS = {"eval", "exec", "open", "compile", "subprocess", "__import__"}
# ALLOWED_IMPORTS = {"torch", "typing"}                           # Added typing for type hints (Tuple, Dict, Tensor)
# REQUIRED_FUNC   = {"compute_reward"}                            # Updated from _get_rewards

# def validate_ast(code: str, required_func: set = REQUIRED_FUNC) -> tuple[bool, str]:
#     try:
#         tree = ast.parse(code)
#     except SyntaxError as e:
#         return False, f"SyntaxError: {e}"

#     func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
#     if not required_func.intersection(func_names):                  # ← THE FIX
#         return False, f"`{required_func}` function not found — got: {func_names}"

#     for node in ast.walk(tree):
#         # Check standard imports (e.g., import torch) and from-imports (e.g., from typing import Tuple)
#         if isinstance(node, (ast.Import, ast.ImportFrom)):
            
#             # Handle 'from module import ...'
#             if hasattr(node, 'module') and node.module is not None:
#                 mod = node.module.split(".")[0]
#                 if mod not in ALLOWED_IMPORTS:
#                     return False, f"Illegal import: {node.module}"
                    
#             # Handle 'import module'
#             for alias in getattr(node, "names", []):
#                 mod = alias.name.split(".")[0]
#                 if mod not in ALLOWED_IMPORTS:
#                     return False, f"Illegal import: {alias.name}"
                    
#         if isinstance(node, ast.Call):
#             func = node.func
#             name = (func.id if isinstance(func, ast.Name) else
#                     func.attr if isinstance(func, ast.Attribute) else None)
#             if name in FORBIDDEN_CALLS:
#                 return False, f"Forbidden call: {name}()"

#     return True, "OK"