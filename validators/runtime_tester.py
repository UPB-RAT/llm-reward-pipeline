# ============================================================
# runtime_tester.py  —  DOMAIN-AGNOSTIC RUNTIME SMOKE TEST
# ============================================================
import re
import inspect
import traceback
import ast

# Assuming torch is available in your runtime environment
import torch

CANDIDATE_FUNC_NAMES = ["_get_rewards"]

_JIT_DECORATOR_RE = re.compile(r"^@torch\.jit\.script\s*\n", re.MULTILINE)

def _strip_jit_decorators(code: str) -> str:
    # We strip the decorator because running JIT compilation inside an exec() 
    # namespace can sometimes cause unnecessary runtime compilation overhead or bugs.
    return _JIT_DECORATOR_RE.sub("", code)

# ── Static Analysis (AST) ───────────────────────────────────────────────────
def _check_for_dim1_on_1d_slices(code: str) -> tuple[bool, str]:
    """
    Statically analyzes the generated code to catch the common mistake of
    applying dim=1 to 1D slices (e.g., pos_error[:, 2]) before execution.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax Error in generated code: {e}"

    for node in ast.walk(tree):
        # Look for function calls (like torch.abs, torch.square, torch.norm, etc.)
        if isinstance(node, ast.Call):
            has_dim_1 = False
            # Check if dim=1 is passed as a keyword argument
            for kw in node.keywords:
                if kw.arg == "dim" and isinstance(kw.value, ast.Constant) and kw.value.value == 1:
                    has_dim_1 = True
            
            if has_dim_1:
                # Check if the first argument is a slice like [:, 2]
                if len(node.args) > 0:
                    first_arg = node.args[0]
                    if isinstance(first_arg, ast.Subscript) and isinstance(first_arg.slice, ast.Tuple):
                        if len(first_arg.slice.elts) == 2:
                            # It's a [:, X] slice
                            return False, (
                                "AST Check Failed: You are applying `dim=1` to a 1D slice. "
                                f"At line {node.lineno}, you passed `dim=1` to an operation on a 1D tensor. "
                                "Remember the rule: Operations on [N] tensors MUST NOT use `dim=1`. "
                                "Remove `dim=1` from this operation."
                            )
    return True, "OK"

# ── Fake Arguments Generator ─────────────────────────────────────────────────
def _make_dummy_args(batch_size: int, domain) -> dict:
    """
    Generates dummy tensors based strictly on the YAML tensor_shapes configuration 
    and packages the cfg_scales into the parameter_dict.
    """
    kwargs = {}
    
    # 1. Create dummy tensors based on shape configs
    for arg_name, shape_list in domain.tensor_shapes.items():
        # Enforce the runtime batch size
        actual_shape = [batch_size] + shape_list[1:]
        kwargs[arg_name] = torch.randn(actual_shape)
        
        # If it's the crashes tensor, clamp to 0 or 1 like the real environment
        if arg_name == "crashes":
            kwargs[arg_name] = torch.where(kwargs[arg_name] > 0, 1.0, 0.0)

    # 2. Add scalar defaults
    kwargs["curriculum_level_multiplier"] = 1.0

    # 3. Create parameter_dict from cfg_scales
    param_dict = {}
    if hasattr(domain, "cfg_scales"):
        for k, v in domain.cfg_scales.items():
            param_dict[k] = torch.tensor(v, dtype=torch.float32)
    kwargs["parameter_dict"] = param_dict

    return kwargs

# ── Main entry point ─────────────────────────────────────────────────────────
def runtime_test(code: str, batch_size: int, domain) -> tuple[bool, str, dict]:
    clean_code = _strip_jit_decorators(code)
    
    # 1. Run static analysis first to catch known LLM hallucinations early
    is_valid_ast, ast_error = _check_for_dim1_on_1d_slices(clean_code)
    if not is_valid_ast:
        return False, ast_error, {}

    # 2. Prepare dummy arguments
    kwargs = _make_dummy_args(batch_size, domain)

    try:
        namespace = {}
        # Execute the generated code string to load the function into namespace
        exec(compile(clean_code, "<reward>", "exec"), {"torch": torch}, namespace)

        # ── Detect function name ──────────────────────────────────────────────
        func_name = next((n for n in CANDIDATE_FUNC_NAMES if n in namespace), None)
        if func_name is None:
            found = [k for k, v in namespace.items() if callable(v) and not k.startswith("__")]
            return False, f"No reward function found. Callable names in code: {found}", {}

        func = namespace[func_name]

        # ── Calling convention ────────────────────────────────────────────────
        # Pass the dictionary as kwargs to the standalone function
        out = func(**kwargs)

        # ── Type & Shape check ────────────────────────────────────────────────
        if not isinstance(out, tuple) or len(out) != 2:
            return False, f"Return type must be a tuple of two items (total_reward, crashes), got {type(out)}", {}

        reward, out_crashes = out

        if not isinstance(reward, torch.Tensor) or not isinstance(out_crashes, torch.Tensor):
            return False, "Both returned items must be torch.Tensor", {}

        if reward.shape != (batch_size,):
            return False, (
                f"Wrong final reward shape: got {tuple(reward.shape)}, expected ({batch_size},). "
                f"You likely forgot to sum over components or used dim=1 incorrectly."
            ), {}
            
        if out_crashes.shape != (batch_size,):
            return False, f"Wrong crashes shape: got {tuple(out_crashes.shape)}, expected ({batch_size},).", {}

        # ── NaN / Inf check ──────────────────────────────────────────────────
        if torch.isnan(reward).any() or torch.isnan(out_crashes).any():
            return False, "Returned tensors contain NaN", {}
        if torch.isinf(reward).any() or torch.isinf(out_crashes).any():
            return False, "Returned tensors contain Inf", {}

        metrics = {
            "mean": reward.mean().item(),
            "std":  reward.std().item(),
            "min":  reward.min().item(),
            "max":  reward.max().item(),
        }
        return True, "OK", metrics

    except Exception as e:
        tb_str = traceback.format_exc()
        error_msg = str(e)
        
        # 2. Enrich the PyTorch error with specific instructions for the LLM
        if "Dimension out of range" in error_msg and "but got 1" in error_msg:
            return False, (
                f"Runtime call failed: {error_msg}\n\n"
                f"CRITICAL FIX NEEDED: You tried to apply `dim=1` on a 1-dimensional tensor (shape [N]). "
                f"Look at your code for operations on things like `[:, 2]` or previously computed 1D variables. "
                f"Remove `dim=1` from these operations. Only use `dim=1` for 2D tensors like `pos_error`."
            ), {}
            
        return False, f"Runtime call failed: {error_msg}\n{tb_str}", {}