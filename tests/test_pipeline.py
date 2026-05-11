from validators.code_extractor import extract_code_block
from validators.ast_validator import static_validate
from validators.runtime_tester import runtime_test

SAMPLE = '''```python
import torch

def compute_reward(uav_pos, uav_vel, uav_ang_vel, uav_quat, goal_pos, prev_dist, crash, episode_len, max_episode_len):
    dist = torch.norm(goal_pos - uav_pos, dim=-1)
    progress = prev_dist - dist
    stability_penalty = 0.05 * torch.norm(uav_ang_vel, dim=-1)
    time_penalty = 0.001 * episode_len.float() / float(max_episode_len)
    goal_bonus = (dist < 1.0).float() * 5.0
    crash_penalty = crash.float() * 10.0
    reward = progress + goal_bonus - stability_penalty - time_penalty - crash_penalty
    return reward.float()
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