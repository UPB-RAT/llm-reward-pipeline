from validators.code_extractor import extract_code_block
from validators.ast_validator import static_validate
from validators.runtime_tester import runtime_test

SAMPLE = '''```python
import torch

def _get_rewards(self) -> torch.Tensor:
    lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
    ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
    distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
    distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
    
    height_deviation = torch.abs(self._robot.data.root_pos_w[:, 2] - self._desired_pos_w[:, 2])
    height_stability = 1.0 - torch.tanh(height_deviation / 0.5)
    
    rewards = {
        "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
        "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
        "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        "height_stability": height_stability * -0.5 * self.step_dt,
    }
    reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
    return reward
```'''


def test_extract_validate_runtime():
    code = extract_code_block(SAMPLE)
    assert code is not None
    ok_static, _ = static_validate(code)
    assert ok_static
    ok_runtime, _, metrics = runtime_test(code)
    assert ok_runtime
    assert "mean" in metrics

# ```

# ## Empty package files

# Use these minimal files:

# ```python
# # prompts/__init__.py
# # reward_generator/__init__.py
# # validators/__init__.py
# # utils/__init__.py
# # tests/__init__.py
# ```

# ## Run commands

# Install and test:

# ```bash
# python -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt
# pytest tests/test_pipeline.py
# ```

# Run generation:

# ```bash
# python -m reward_generator.cli \
#   --model-path models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
#   --num-candidates 5 \
#   --task long_range_navigation
# ```

# ## What this gives you

# This repo gives you a clean common base for Phase 1: a local code model, a UAV reward prompt contract, a static validator, and a runtime smoke test. That is the right foundation before attaching IsaacLab rollout evaluation, because reward generation quality depends heavily on prompt structure and validation discipline.[4][5][1]

# The next logical step is to improve this base with three additions: structured scoring of accepted rewards, templated prompt variants for exploration, and a small benchmark set of hand-written reward examples for regression checks. IsaacLab’s task workflow and IsaacLabEureka make that Phase 2 bridge straightforward once this repo is stable.[2][3]

# Would you like me to produce **Version 2** of this repo with:
# 1. Dockerfile,
# 2. `pytest` suite expanded,
# 3. ranking/scoring module,
# 4. prompt variants,
# 5. IsaacLab integration stubs for Phase 2?