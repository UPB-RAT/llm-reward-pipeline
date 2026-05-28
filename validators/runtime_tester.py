# ============================================================
# runtime_tester.py  —  DOMAIN-AGNOSTIC RUNTIME SMOKE TEST
# ============================================================
import re
import types
import inspect
import traceback
import ast

# Assuming torch is available in your runtime environment
import torch

CANDIDATE_FUNC_NAMES = ["_get_rewards"]

_JIT_DECORATOR_RE = re.compile(r"^@torch\.jit\.script\s*\n", re.MULTILINE)

def _strip_jit_decorators(code: str) -> str:
    return _JIT_DECORATOR_RE.sub("", code)

def _is_method_style(func) -> bool:
    params = list(inspect.signature(func).parameters.keys())
    return len(params) > 0 and params[0] == "self"

# ── Static Analysis (AST) ───────────────────────────────────────────────────
def _check_for_dim1_on_1d_slices(code: str) -> tuple[bool, str]:
    """
    Statically analyzes the generated code to catch the common mistake of
    applying dim=1 to 1D slices (e.g., self._robot.data.root_pos_w[:, 2])
    before execution.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax Error in generated code: {e}"

    for node in ast.walk(tree):
        # Look for function calls (like torch.abs, torch.square, etc.)
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

# ── Fake env ─────────────────────────────────────────────────────────────────
def _make_dummy_env(batch_size: int, domain) -> types.SimpleNamespace:
    """
    Covers all attribute access patterns the LLM may use.
    """
    env = types.SimpleNamespace()

    cfg = types.SimpleNamespace(**domain.cfg_scales)
    env.cfg = cfg
    env.step_dt = domain.step_dt

    # ── Quadcopter-style nested robot attributes ──────────────────────────────
    robot_data = types.SimpleNamespace()
    desired_pos = torch.randn(batch_size, 3)
    root_pos    = torch.randn(batch_size, 3)
    root_pos[:, 2] = torch.rand(batch_size) * 1.5 + 0.3

    for attr, shape in domain.tensor_shapes.items():
        parts = attr.split(".")
        if len(parts) >= 3 and parts[0] in ("robot", "_robot"):
            tensor = torch.randn([batch_size] + shape[1:])
            setattr(robot_data, parts[-1], tensor)

    robot = types.SimpleNamespace(data=robot_data)
    env.robot  = robot
    env._robot = robot

    env.desired_pos_w  = desired_pos
    env._desired_pos_w = desired_pos
    env.robot.data.root_pos_w  = root_pos
    env._robot.data.root_pos_w = root_pos

    # ── Cartpole-style attributes ─────────────────────────────────────────────
    joint_pos = torch.randn(batch_size, 2)
    joint_vel = torch.randn(batch_size, 2)
    env.joint_pos = joint_pos
    env.joint_vel = joint_vel

    env._pole_dof_idx = [1]
    env._cart_dof_idx = [0]

    env.reset_terminated = (torch.rand(batch_size) > 0.9)
    env.reset_buf        = env.reset_terminated
    env.terminated       = env.reset_terminated

    return env

# ── Main entry point ─────────────────────────────────────────────────────────
def runtime_test(code: str, batch_size: int, domain) -> tuple[bool, str, dict]:
    clean_code = _strip_jit_decorators(code)
    
    # 1. Run static analysis first to catch known LLM hallucinations early
    is_valid_ast, ast_error = _check_for_dim1_on_1d_slices(clean_code)
    if not is_valid_ast:
        return False, ast_error, {}

    env = _make_dummy_env(batch_size, domain)

    try:
        namespace = {}
        exec(compile(clean_code, "<reward>", "exec"), {"torch": torch}, namespace)

        # ── Detect function name ──────────────────────────────────────────────
        func_name = next((n for n in CANDIDATE_FUNC_NAMES if n in namespace), None)
        if func_name is None:
            found = [k for k, v in namespace.items() if callable(v) and not k.startswith("__")]
            return False, f"No reward function found. Callable names in code: {found}", {}

        func = namespace[func_name]

        # ── Calling convention ────────────────────────────────────────────────
        if _is_method_style(func):
            bound  = types.MethodType(func, env)
            reward = bound()
        else:
            return False, "Function must take `self` as the first argument.", {}

        # ── Shape check ──────────────────────────────────────────────────────
        if not isinstance(reward, torch.Tensor):
            return False, f"Return type is {type(reward).__name__}, expected torch.Tensor", {}
        if reward.shape != (batch_size,):
            return False, (
                f"Wrong final shape: got {tuple(reward.shape)}, expected ({batch_size},). "
                f"You likely forgot to sum over components or used dim=1 incorrectly."
            ), {}

        # ── NaN / Inf check ──────────────────────────────────────────────────
        if torch.isnan(reward).any():
            return False, "Reward contains NaN", {}
        if torch.isinf(reward).any():
            return False, "Reward contains Inf", {}

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
                f"Remove `dim=1` from these operations. Only use `dim=1` for 2D tensors like `root_pos_w`."
            ), {}
            
        return False, f"Runtime call failed: {error_msg}\n{tb_str}", {}