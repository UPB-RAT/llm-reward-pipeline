import torch
from validators.code_extractor import extract_code
from validators.ast_validator import validate_ast
from validators.runtime_tester import runtime_test
from reward_generator.domain_config import load_config


# Load real domain config so tests reflect actual pipeline settings
CFG = load_config("configs/default.yaml")


SAMPLE = '''```python
import torch

def _get_rewards(self) -> torch.Tensor:
    lin_vel = torch.sum(torch.square(self.robot.data.root_lin_vel_b), dim=1)
    ang_vel = torch.sum(torch.square(self.robot.data.root_ang_vel_b), dim=1)
    distance_to_goal = torch.linalg.norm(
        self.desired_pos_w - self.robot.data.root_pos_w, dim=1
    )
    distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
    height_deviation = torch.abs(
        self.robot.data.root_pos_w[:, 2] - self.desired_pos_w[:, 2]
    )
    rewards = {
        "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
        "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
        "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        "height_stability": height_deviation * -1.0 * self.step_dt,
    }
    reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
    return reward
```'''


SAMPLE_NO_BLOCK = "Here is some text without a code block."


SAMPLE_WRONG_FUNC = '''```python
def compute_reward(self):
    return torch.zeros(16)
```'''


SAMPLE_FORBIDDEN = '''```python
def _get_rewards(self):
    exec("malicious code")
    return torch.zeros(16)
```'''


# ── Stage 1: Code Extraction ─────────────────────────────────────────────────

def test_extract_valid_block():
    code = extract_code(SAMPLE)
    assert code is not None
    assert "_get_rewards" in code


def test_extract_no_block():
    code = extract_code(SAMPLE_NO_BLOCK)
    assert code is None


# ── Stage 2: AST Static Validation ───────────────────────────────────────────

def test_static_validate_pass():
    code = extract_code(SAMPLE)
    ok, msg = validate_ast(code)
    assert ok, f"Expected pass but got: {msg}"


def test_static_validate_wrong_function_name():
    code = extract_code(SAMPLE_WRONG_FUNC)
    ok, _ = validate_ast(code)
    assert not ok


def test_static_validate_forbidden_call():
    code = extract_code(SAMPLE_FORBIDDEN)
    ok, msg = validate_ast(code)
    assert not ok
    assert "exec" in msg.lower()


# ── Stage 4: Runtime Smoke Test ──────────────────────────────────────────────

def test_runtime_pass():
    code = extract_code(SAMPLE)
    ok, msg, metrics = runtime_test(
        code,
        batch_size=CFG.runtime_test.batch_size,
        domain=CFG.domain,
    )
    assert ok, f"Expected runtime pass but got: {msg}"
    assert "mean" in metrics
    assert "std"  in metrics


def test_runtime_wrong_shape():
    bad_code = '''
def _get_rewards(self) -> torch.Tensor:
    import torch
    return torch.zeros(1)   # wrong shape — not (N,)
'''
    ok, msg, _ = runtime_test(
        bad_code,
        batch_size=CFG.runtime_test.batch_size,
        domain=CFG.domain,
    )
    assert not ok
    assert "shape" in msg.lower()