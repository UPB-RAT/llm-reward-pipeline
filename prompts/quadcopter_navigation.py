# 2. Define the Updated Prompts
SYSTEM_PROMPT = """\
You are an expert robotics engineer writing compact PyTorch reward functions for an Isaac Lab quadcopter simulation.

CRITICAL INSTRUCTIONS - Violating any rule means your output is discarded:

<format_rules>
- Output ONLY a single python function named exactly: `def _get_rewards(self) -> torch.Tensor:`
- Wrap your answer in a single ```python ... ``` block. Do not output ANY text, explanations, or comments outside or inside the block.
- Maximum 15 lines of code inside the function body.
</format_rules>

<coding_constraints>
- RETURN TYPE: Must return a 1-D float32 tensor of shape [N] matching the batch size.
- NO IMPORTS: `torch` is already available. Do not use import, eval, exec, or open.
- TENSOR SHAPES & DIMENSIONS (CRITICAL RULE):
  * Operations on [N, 3] tensors MUST use `dim=1`.
  * Operations on [N] tensors MUST NOT use `dim=1`.
- ATTRIBUTES: Use ONLY the attributes strictly listed in the <domain_constraints> below.
- Do not reference gate normals, plane-crossing events, success flags, previous-step buffers, collision signals, episode progress, or sequence indices unless they are explicitly listed in <domain_constraints>.
</coding_constraints>

<task_interpretation>
- Interpret `self._desired_pos_w` as the current target waypoint / gate center provided by the environment.
- Because gate orientation and pass-through events are not available, approximate the task with dense shaping that helps the quadcopter reach each new target quickly and stably.
- Favor rewards that encourage: moving toward the target, being close to the target, maintaining useful forward speed toward the target, and staying dynamically stable.
- Favor penalties that discourage: sideways or wasteful velocity, excessive angular velocity, and excessive tilt.
- Do not invent sparse completion bonuses that depend on unavailable event signals.
</task_interpretation>

<required_structure>
1. Define local reward scale variables at the top.
2. Compute each reward component as a named variable of shape [N].
3. Add at least 2 novel components beyond plain linear velocity, angular velocity, and distance-to-goal.
4. Collect all computed components into a dictionary called `rewards`.
5. Return exactly: `torch.sum(torch.stack(list(rewards.values())), dim=0)`
</required_structure>
"""

task_description = """
We train a quadrotor in an Isaac Lab RL environment to fly aggressively through a sequential curriculum of targets.
At any instant, `self._desired_pos_w` represents the center of the current gate / waypoint selected by the environment's goal generator.
When the robot reaches or passes the current gate, the environment switches `self._desired_pos_w` to the next target in the sequence.
The sequence length varies across episodes.

Reward intent:
- Encourage reaching the current target as quickly as possible.
- Encourage velocity that is aligned with the direction from the robot to the current target.
- Encourage useful speed toward the target, not speed in arbitrary directions.
- Penalize excessive angular velocity and excessive tilt to preserve controllability.
- Penalize overall motion that is wasteful or unstable when it does not help target-reaching.
- Since gate orientation, gate-plane crossing tests, and miss/success indicators are not available in the observation tensors, do not use such terms directly.
"""

TENSOR_REFERENCE = """\
<domain_constraints>
Available attributes you can use:
    pos_error, crashes, action, prev_action, curriculum_level_multiplier, parameter_dict
</domain_constraints>
"""

FEW_SHOT_EXAMPLE = """\
<valid_example>
```python
def _get_rewards(
    pos_error, crashes, action, prev_action, curriculum_level_multiplier, parameter_dict
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Dict[str, Tensor]) -> Tuple[Tensor, Tensor]
    dist = torch.norm(pos_error, dim=1)

    pos_reward = 2.0 / (1.0 + dist * dist)

    dist_reward = (20 - dist) / 20.0

    total_reward = (
        pos_reward
        + dist_reward  # + up_reward + action_diff_reward + absolute_action_reward
    )
    total_reward[:] = curriculum_level_multiplier * total_reward
    crashes[:] = torch.where(dist > 8.0, torch.ones_like(crashes), crashes)

    total_reward[:] = torch.where(
        crashes > 0.0, -2 * torch.ones_like(total_reward), total_reward
    )
    return total_reward, crashes
```
</valid_example>
"""

USER_TEMPLATE = f"""\
TASK DESCRIPTION:
{task_description}

{TENSOR_REFERENCE}

{FEW_SHOT_EXAMPLE}

BASELINE TO EXTEND (Do NOT copy exactly. Max 15 lines, no comments, add at least 2 novel components):
```python
def _get_rewards(
    pos_error, crashes, action, prev_action, curriculum_level_multiplier, parameter_dict
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Dict[str, Tensor]) -> Tuple[Tensor, Tensor]
    dist = torch.norm(pos_error, dim=1)

    pos_reward = 2.0 / (1.0 + dist * dist)

    dist_reward = (20 - dist) / 20.0

    total_reward = (
        pos_reward
        + dist_reward  # + up_reward + action_diff_reward + absolute_action_reward
    )
    total_reward[:] = curriculum_level_multiplier * total_reward
    crashes[:] = torch.where(dist > 8.0, torch.ones_like(crashes), crashes)

    total_reward[:] = torch.where(
        crashes > 0.0, -2 * torch.ones_like(total_reward), total_reward
    )
    return total_reward, crashes
```

TASK: Write the final reward function for variant: Agile Target Reaching.
Prefer dense shaping terms such as target-distance mapping, velocity-direction alignment, speed-toward-target, lateral-velocity penalty, tilt penalty, and smooth stability-oriented penalties that only use allowed tensors.
Remember: Output ONLY the raw python code block. Do not write any explanations.
"""