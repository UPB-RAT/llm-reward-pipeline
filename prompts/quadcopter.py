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
  * Operations on [N, 3] tensors MUST use `dim=1` (e.g., `torch.sum(tensor_2d, dim=1)`).
  * Operations on [N] tensors (like 1D slices `self._robot.data.root_pos_w[:, 2]`) MUST NOT use `dim=1`.
  * COMMON MISTAKE TO AVOID: Do NOT write `torch.abs(self._robot.data.root_pos_w[:, 2] - self._desired_pos_w[:, 2], dim=1)`. This will crash with "Dimension out of range". Write `torch.abs(self._robot.data.root_pos_w[:, 2] - self._desired_pos_w[:, 2])` instead.
- ATTRIBUTES: Use ONLY the attributes strictly listed in the <domain_constraints> section of the task.
</coding_constraints>

<required_structure>
You MUST use this exact mathematical pattern for your function:
1. Define any local reward scale variables you need at the top.
2. Compute each reward component as a named variable of shape [N].
3. Add at least 2 novel components not present in the example above.
4. Collect all computed components into a dictionary called `rewards`.
5. Return exactly: `torch.sum(torch.stack(list(rewards.values())), dim=0)`
</required_structure>
"""

TENSOR_REFERENCE = """\
<domain_constraints>
Available attributes you can use:
- `self._robot.data.root_pos_w` : [N, 3] tensor, quadcopter global position
- `self._robot.data.root_lin_vel_b` : [N, 3] tensor, quadcopter base linear velocity
- `self._robot.data.root_ang_vel_b` : [N, 3] tensor, quadcopter base angular velocity 
- `self._robot.data.projected_gravity_b` : [N, 3] tensor, gravity vector projected in base frame
- `self._desired_pos_w` : [N, 3] tensor, goal global position
- `self.step_dt` : float, time step duration
</domain_constraints>
"""

FEW_SHOT_EXAMPLE = """\
<valid_example>
```python
def _get_rewards(self) -> torch.Tensor:
    # 1. Define scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    tilt_reward_scale = -0.05
    
    # 2. Compute metrics
    lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
    ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
    distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
    tilt = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
    
    # 3. Assemble dictionary
    rewards = {
        "lin_vel": lin_vel * lin_vel_reward_scale * self.step_dt,
        "ang_vel": ang_vel * ang_vel_reward_scale * self.step_dt,
        "distance_to_goal": (1 - torch.tanh(distance_to_goal / 0.8)) * distance_to_goal_reward_scale * self.step_dt,
        "tilt": tilt * tilt_reward_scale * self.step_dt,
    }
    
    # 4. Return
    return torch.sum(torch.stack(list(rewards.values())), dim=0)
```
</valid_example>

UNIQUENESS REQUIREMENT:
The example above is for illustration only - DO NOT copy it. Your function MUST be structurally different: use different component names, different shaping functions, and different scale constants. Repeating the example is grounds for rejection.
"""

USER_TEMPLATE = """\
TASK DESCRIPTION:
{task_description}

{tensor_reference}

{few_shot}

{prior_results}

TASK: Write the final reward function for variant: {variant}.
Remember: Output ONLY the raw python code block. Do not write any explanations.
"""
