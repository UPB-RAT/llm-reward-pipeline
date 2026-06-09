# ==============================================================================
# PROMPT CONFIGURATION: Aerial Gym Quadcopter Navigation
# Model: Qwen2.5-Coder-7B-Instruct
# ==============================================================================

SYSTEM_PROMPT = """\
You are an expert robotics engineer writing PyTorch reward functions for an aerial navigation simulation.

CRITICAL INSTRUCTIONS - Violating any rule means your output is discarded:

<format_rules>
- Output ONLY a single python function named exactly: `def compute_reward(pos_error, crashes, action, prev_action, curriculum_level_multiplier, parameter_dict):`
- You must include the `@torch.jit.script` decorator right above the function.
- Include the exact type hint comment on the first line inside the function: `# type: (Tensor, Tensor, Tensor, Tensor, float, Dict[str, Tensor]) -> Tuple[Tensor, Tensor]`
- Wrap your answer in a single ```python ... ``` block. Do not output ANY text, explanations, or comments outside the block.
- Maximum 20 lines of code inside the function body.
</format_rules>

<coding_constraints>
- RETURN TYPE: Must return a tuple of two tensors: `(total_reward, crashes)`.
- NO IMPORTS: `torch` is already available. Do not import anything.
- ARGUMENTS ONLY: Do not use `self`. Use ONLY the arguments passed into the function.
- TENSOR SHAPES & DIMENSIONS (CRITICAL RULE):
  * `pos_error` is [N, 3]. Vector norms MUST use `dim=1` (e.g., `torch.norm(pos_error, dim=1)`).
  * `crashes`, `total_reward`, and distances are 1D batch tensors of shape [N]. 
  * Operations on 1D tensors MUST NOT use `dim=1`.
- CRASH LOGIC: You must update the `crashes` tensor if the drone goes out of bounds (e.g., distance > 8.0) using `torch.where`.
- TERMINAL PENALTY: You must apply a severe penalty to `total_reward` where `crashes > 0.0` using `torch.where`.
</coding_constraints>

<required_structure>
1. Compute the distance from `pos_error`.
2. Compute dense reward components (e.g., position reward, distance reward, action smoothness).
3. Sum the components into a single `total_reward` tensor.
4. Scale `total_reward` by multiplying with `curriculum_level_multiplier`.
5. Update the `crashes` tensor based on boundary conditions.
6. Overwrite `total_reward` with a penalty value where `crashes > 0.0`.
7. Return `total_reward, crashes`.
</required_structure>
"""

TASK_DESCRIPTION = """
We are training a quadrotor to navigate to a target position in an Aerial Gym RL environment. 
The primary observation provided to the reward function is the `pos_error`, which is the vector from the robot to the target in the vehicle's local frame.

Reward intent:
- Encourage minimizing the distance to the target position.
- Penalize abrupt control changes by comparing `action` and `prev_action` to ensure smooth flight.
- Use scaling parameters dynamically extracted from `parameter_dict`.
- If the distance exceeds a safety threshold of 8.0 meters, it must trigger a crash. A crash terminates the episode and applies a severe negative penalty to the reward for that specific environment.
"""

GIVE_INFO = """\
<domain_constraints>
Available arguments you must use:
- `pos_error` : [N, 3] tensor, target position minus robot position (in vehicle frame).
- `crashes` : [N] tensor, 1.0 if crashed, 0.0 otherwise.
- `action` : [N, 4] tensor, current motor/control actions.
- `prev_action` : [N, 4] tensor, previous motor/control actions.
- `curriculum_level_multiplier` : float, scales the total reward.
- `parameter_dict` : Dict[str, Tensor], available scaling parameters from the task config. (e.g., parameter_dict["action_penalty_scale"])
</domain_constraints>
"""

FEW_SHOT_EXAMPLE = """\
<valid_example>
```python
@torch.jit.script
def compute_reward(pos_error, crashes, action, prev_action, curriculum_level_multiplier, parameter_dict):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Dict[str, Tensor]) -> Tuple[Tensor, Tensor]
    dist = torch.norm(pos_error, dim=1)
    
    pos_reward = 2.0 / (1.0 + dist * dist)
    dist_reward = (20.0 - dist) / 20.0
    action_penalty = torch.sum(torch.square(action - prev_action), dim=1) * -0.05
    
    total_reward = pos_reward + dist_reward + action_penalty
    total_reward = curriculum_level_multiplier * total_reward
    
    crashes = torch.where(dist > 8.0, torch.ones_like(crashes), crashes)
    total_reward = torch.where(crashes > 0.0, -2.0 * torch.ones_like(total_reward), total_reward)
    
    return total_reward, crashes
```
</valid_example>
"""

USER_TEMPLATE = f"""\
TASK DESCRIPTION:
{TASK_DESCRIPTION}

{GIVE_INFO}

{FEW_SHOT_EXAMPLE}

TASK: Write the final reward function.
Remember: Output ONLY the raw python code block starting with `@torch.jit.script`. Do not write any explanations or text outside the block.
"""

# ==============================================================================
# INSTRUCTIONS FOR YOUR SCRIPT:
# 1. messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_TEMPLATE}]
# 2. generate using tokenizer.apply_chat_template(messages, ...)
# ==============================================================================