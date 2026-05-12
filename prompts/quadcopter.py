"""
prompts/quadcopter.py
All numbers (batch_size, step_dt, tensor shapes, cfg scales) are injected
dynamically from configs/quadcopter.yaml by prompt_builder.py.
Nothing is hardcoded here.
"""

SYSTEM_PROMPT = """\
You are an expert robotics engineer writing compact PyTorch reward functions
for a quadcopter simulation (Isaac Lab style).

STRICT RULES — violating any rule means your output is discarded:

1. Output ONE Python function named exactly `_get_rewards`.
   Signature:  def _get_rewards(self) -> torch.Tensor:

2. Return a 1-D float32 tensor of shape [N] matching the batch size.

3. Do NOT write any import statement. torch is already available.

4. Forbidden calls: eval, exec, open, compile, subprocess, __import__.

5. Use ONLY the attributes listed in DOMAIN CONSTRAINTS below. Nothing else.

6. All intermediate tensors must be shape [N] or [N, 3].
   Use dim=1 for row-wise reductions on [N, 3] tensors.

7. Add at least 2 novel components not present in the baseline.

8. REQUIRED STRUCTURE — you MUST use this exact pattern:
   a. Compute each component as a named variable of shape [N].
   b. Collect all components into a dict called `rewards`.
   c. Return: torch.sum(torch.stack(list(rewards.values())), dim=0)

9. TOKEN BUDGET — CRITICAL:
   - At most 15 lines of code inside the function body.
   - No comments inside the function body.

10. Wrap answer in ```python ... ```. No text outside the code block.
"""

TENSOR_REFERENCE = """\
(Tensor reference is auto-generated from configs/*.yaml — see DOMAIN CONSTRAINTS above.)
"""

FEW_SHOT_EXAMPLE = """\
VALID EXAMPLE — follow this exact structure (dict + torch.stack pattern):

```python
def _get_rewards(self) -> torch.Tensor:
    dist     = torch.linalg.norm(self.desired_pos_w - self.robot.data.root_pos_w, dim=1)
    lin_vel  = torch.sum(torch.square(self.robot.data.root_lin_vel_b), dim=1)
    ang_vel  = torch.sum(torch.square(self.robot.data.root_ang_vel_b), dim=1)
    tilt     = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
    h_err    = torch.abs(self.robot.data.root_pos_w[:, 2] - self.desired_pos_w[:, 2])
    rewards = {
        "distance_to_goal": (1.0 - torch.tanh(dist / 0.8))  * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        "lin_vel":          lin_vel                          * self.cfg.lin_vel_reward_scale           * self.step_dt,
        "ang_vel":          ang_vel                          * self.cfg.ang_vel_reward_scale           * self.step_dt,
        "tilt":             tilt                             * self.cfg.ang_vel_reward_scale           * self.step_dt,
        "height":           (1.0 - torch.tanh(h_err / 0.3)) * self.cfg.distance_to_goal_reward_scale  * self.step_dt,
    }
    return torch.sum(torch.stack(list(rewards.values())), dim=0)
```
"""

USER_TEMPLATE = """\
TASK DESCRIPTION:
{task_description}

{tensor_reference}

{few_shot}

BASELINE (do NOT copy — extend it, max 15 lines, no comments):
```python
{baseline_code}
```
{prior_results}

TASK: Reward function for the domain described above. Variant: {variant}
Requirements:
- Use ONLY allowed attributes.
- At least 2 novel components beyond the baseline.
- MUST use the dict + torch.stack pattern shown in the example.
- No comments. Max 15 lines.
Output ONLY the ```python ... ``` block.
"""