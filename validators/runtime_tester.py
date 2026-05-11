# ============================================================
# runtime_tester.py  —  DOMAIN-AGNOSTIC RUNTIME SMOKE TEST
# ============================================================
import types
import torch
import traceback


def _make_dummy_env(batch_size: int, domain) -> types.SimpleNamespace:
    """Build a fake self with all tensors the reward function may access."""
    env = types.SimpleNamespace()

    # ── Tensors from domain config ────────────────────────────────
    tensors = {}
    for attr, shape in domain.tensor_shapes.items():
        full_shape = [batch_size] + shape[1:]  # replace N with batch_size
        tensors[attr] = torch.randn(full_shape)

    # ── cfg scales ───────────────────────────────────────────────
    cfg = types.SimpleNamespace(**domain.cfg_scales)
    env.cfg = cfg

    # ── step_dt ──────────────────────────────────────────────────
    env.step_dt = domain.step_dt

    # ── Build nested attribute access: robot.data.root_lin_vel_b ─
    # Supports BOTH self.robot.* AND self._robot.* (LLM uses both)
    robot_data = types.SimpleNamespace()
    desired_pos = torch.randn(batch_size, 3)
    root_pos    = torch.randn(batch_size, 3)
    root_pos[:, 2] = torch.rand(batch_size) * 1.5 + 0.3  # Z in [0.3, 1.8]

    for attr, shape in domain.tensor_shapes.items():
        # attr looks like "robot.data.root_lin_vel_b"
        parts = attr.split(".")
        if len(parts) >= 3 and parts[0] in ("robot", "_robot"):
            tensor = torch.randn([batch_size] + shape[1:])
            setattr(robot_data, parts[-1], tensor)

    robot = types.SimpleNamespace(data=robot_data)

    # Expose as both self.robot and self._robot
    env.robot  = robot
    env._robot = robot

    # Expose desired_pos as both variants
    env.desired_pos_w  = desired_pos
    env._desired_pos_w = desired_pos

    # Expose root_pos via robot.data too
    env.robot.data.root_pos_w  = root_pos
    env._robot.data.root_pos_w = root_pos

    return env


def runtime_test(code: str, batch_size: int, domain) -> tuple[bool, str, dict]:
    env = _make_dummy_env(batch_size, domain)

    try:
        namespace = {}
        exec(compile(code, "<reward>", "exec"), {"torch": torch}, namespace)

        func = namespace.get("_get_rewards")
        if func is None:
            return False, "_get_rewards not found in code", {}

        import types as _types
        bound = _types.MethodType(func, env)
        reward = bound()

        # ── Shape check ──────────────────────────────────────────
        if not isinstance(reward, torch.Tensor):
            return False, f"Return type is {type(reward)}, expected torch.Tensor", {}
        if reward.shape != (batch_size,):
            return False, f"Wrong shape: {tuple(reward.shape)}, expected ({batch_size},)", {}

        # ── NaN / Inf check ──────────────────────────────────────
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
        return False, f"Runtime call failed: {e}", {}