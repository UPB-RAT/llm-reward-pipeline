import torch


def runtime_test(code: str, batch_size: int = 16, max_episode_len: int = 500) -> tuple[bool, str, dict]:
    torch.manual_seed(42)
    N = batch_size
    local_ns = {"torch": torch}

    try:
        exec(code, local_ns)
    except Exception as e:
        return False, f"exec failed: {e}", {}

    fn = local_ns.get("_get_rewards") or local_ns.get("compute_reward")
    if fn is None:
        return False, "Neither '_get_rewards' nor 'compute_reward' found after exec", {}

    dummy_env = _build_dummy_env(N, max_episode_len)

    # Patch rewards dict to intercept per-component shapes before stack
    dummy_env._shape_errors = []
    original_dict = dict

    try:
        reward = fn(dummy_env)
    except Exception as e:
        # Try to give a helpful shape diagnosis
        msg = _diagnose_shape_error(str(e), fn, dummy_env, N)
        return False, msg, {}

    return _validate_output(reward, N)


def _diagnose_shape_error(error_msg: str, fn, dummy_env, N: int) -> str:
    """Re-run with monkey-patched rewards dict to find the bad component."""
    import types

    shape_log = {}
    original_setitem = dict.__setitem__

    class TrackedDict(dict):
        def __setitem__(self, key, value):
            if isinstance(value, torch.Tensor):
                shape_log[key] = tuple(value.shape)
            super().__setitem__(key, value)

    # Patch builtins to use TrackedDict
    import builtins
    original_dict_builtin = builtins.__dict__.get("dict")

    try:
        local_ns2 = {"torch": torch, "dict": TrackedDict}
        exec(fn.__code__, local_ns2) if hasattr(fn, "__code__") else None
    except Exception:
        pass

    bad = {k: v for k, v in shape_log.items() if v != (N,)}
    if bad:
        details = ", ".join(f"'{k}': shape={v}" for k, v in bad.items())
        return (
            f"Shape mismatch in reward components — expected ({N},) for all. "
            f"Bad components: {details}. "
            f"Likely cause: .mean() or boolean indexing collapsed shape. "
            f"Use `tensor * condition` instead of `tensor[condition]`."
        )
    return f"Runtime call failed: {error_msg}"


def _build_dummy_env(N: int, max_episode_len: int):
    class _Data:
        pass

    class _Robot:
        def __init__(self):
            self.data = _Data()
            self.data.root_lin_vel_b          = torch.randn(N, 3)
            self.data.root_ang_vel_b          = torch.randn(N, 3)
            self.data.root_pos_w              = torch.randn(N, 3)
            self.data.projected_gravity_b     = torch.randn(N, 3)

    class _Cfg:
        lin_vel_reward_scale              = -0.05
        ang_vel_reward_scale              = -0.05
        distance_to_goal_reward_scale     = 15.0
        # New component scales the LLM may introduce
        height_stability_reward_scale     = -1.0
        smooth_approach_reward_scale      = -0.5
        gravity_alignment_reward_scale    = -0.5
        crash_penalty_scale               = -10.0
        progress_reward_scale             = 10.0
        goal_bonus_scale                  = 5.0
        energy_penalty_scale              = -0.01

        def __getattr__(self, name):
            # Catch-all: any unknown cfg attribute returns a safe default
            # instead of raising AttributeError
            if "scale" in name:
                return -1.0 if "penalty" in name or "vel" in name else 1.0
            raise AttributeError(f"'_Cfg' object has no attribute '{name}'")

    class _DummyEnv:
        def __init__(self):
            self._robot         = _Robot()
            self._desired_pos_w = torch.randn(N, 3) * 10.0
            self.cfg            = _Cfg()
            self.step_dt        = 0.02
            self.episode_length_buf = torch.randint(0, max_episode_len, (N,))

    return _DummyEnv()


def _validate_output(reward, N: int) -> tuple[bool, str, dict]:
    if not isinstance(reward, torch.Tensor):
        return False, f"Output is not a torch.Tensor (got {type(reward)})", {}
    if reward.shape != (N,):
        return False, f"Wrong output shape: {tuple(reward.shape)}, expected ({N},)", {}
    if torch.isnan(reward).any():
        return False, "NaN in reward output", {}
    if torch.isinf(reward).any():
        return False, "Inf in reward output", {}
    metrics = {
        "min":  float(reward.min()),
        "max":  float(reward.max()),
        "mean": float(reward.mean()),
        "std":  float(reward.std()),
    }
    return True, "Passed runtime smoke test", metrics