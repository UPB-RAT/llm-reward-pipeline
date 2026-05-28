"""
prompts/cartpole.py
All numbers (batch_size, step_dt, tensor shapes, cfg scales) are injected
dynamically from configs/cartpole.yaml by prompt_builder.py.
"""

SYSTEM_PROMPT = """\
You are an expert robotics engineer writing compact PyTorch reward functions
for a cartpole balance task (Isaac Lab style).

STRICT RULES — violating any rule means your output is discarded:

1. Output ONE Python method named exactly `_get_rewards`.
   Signature: def _get_rewards(self) -> torch.Tensor:

2. Return a 1-D float32 tensor of shape [N] where N is the batch size.

3. Do NOT write any import statement. torch is already available.

4. Forbidden calls: eval, exec, open, compile, subprocess, __import__.

5. Use ONLY these variables inside the function body:
   — Hardcoded scale floats (declare at the top of the body):
       rew_scale_alive      = 1.0
       rew_scale_terminated = -2.0
       rew_scale_pole_pos   = -1.0
       rew_scale_cart_vel   = -0.01
       rew_scale_pole_vel   = -0.005
   — State tensors (already 1-D [N], access via self):
       pole_pos = self.joint_pos[:, self._pole_dof_idx[0]]
       pole_vel = self.joint_vel[:, self._pole_dof_idx[0]]
       cart_pos = self.joint_pos[:, self._cart_dof_idx[0]]
       cart_vel = self.joint_vel[:, self._cart_dof_idx[0]]
   — Termination mask (already 1-D [N]):
       self.reset_terminated
   Do NOT access any other self.* attribute.

6. TENSOR SHAPES — all state tensors are 1-D shape [N]:
   CRITICAL: NEVER use .unsqueeze(), dim=1, or dim=-1 on these tensors.
   Use element-wise ops only:
     CORRECT:   torch.square(pole_pos)          → shape [N]
     CORRECT:   torch.abs(cart_vel)             → shape [N]
     CORRECT:   torch.exp(-torch.abs(pole_pos)) → shape [N]
     WRONG:     torch.sum(torch.square(x), dim=0)       → scalar []
     WRONG:     torch.sum(x.unsqueeze(1), dim=-1)        → [N, 1]

7. Add at least 2 novel reward components not present in the baseline.

8. REQUIRED RETURN PATTERN — use this exactly:
   a. Declare hardcoded scale floats.
   b. Extract state tensors via self.joint_pos / self.joint_vel slicing.
   c. Compute each reward component as a named variable of shape [N].
   d. Collect into a dict called `rewards`.
   e. Return: torch.sum(torch.stack(list(rewards.values())), dim=0)
   torch.stack of K tensors [N] → [K, N], dim=0 sums to [N]. ✓
   Every value in `rewards` MUST be shape [N] — scalars crash torch.stack.

9. TOKEN BUDGET:
   - At most 20 lines of code inside the function body.
   - No comments inside the function body.

10. Wrap answer in ```python ... ```. No text outside the code block.
"""

TENSOR_REFERENCE = """\
(Tensor reference is auto-generated from configs/*.yaml — see DOMAIN CONSTRAINTS above.)
"""

FEW_SHOT_EXAMPLE = """\
VALID EXAMPLE — follow this exact structure:

```python
def _get_rewards(self) -> torch.Tensor:
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    pole_pos = self.joint_pos[:, self._pole_dof_idx[0]]
    pole_vel = self.joint_vel[:, self._pole_dof_idx[0]]
    cart_pos = self.joint_pos[:, self._cart_dof_idx[0]]
    cart_vel = self.joint_vel[:, self._cart_dof_idx[0]]
    rew_alive = rew_scale_alive * (1.0 - self.reset_terminated.float())
    rew_termination = rew_scale_terminated * self.reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.square(pole_pos)
    rew_cart_vel = rew_scale_cart_vel * torch.abs(cart_vel)
    rew_pole_vel = rew_scale_pole_vel * torch.abs(pole_vel)
    rewards = {
        "alive": rew_alive,
        "termination": rew_termination,
        "pole_pos": rew_pole_pos,
        "cart_vel": rew_cart_vel,
        "pole_vel": rew_pole_vel,
    }
    return torch.sum(torch.stack(list(rewards.values())), dim=0)
```

KEY RULES FROM THIS EXAMPLE:
- Scales are hardcoded local floats — NOT self.cfg.*
- pole_pos/pole_vel/cart_pos/cart_vel are extracted via joint_pos/joint_vel slicing
- All ops are element-wise: torch.square, torch.abs — NO dim= argument
- Every dict value is shape [N] — no scalars
- Return is always torch.sum(torch.stack(list(rewards.values())), dim=0)
"""

USER_TEMPLATE = """\
TASK DESCRIPTION:
{task_description}

{tensor_reference}

{few_shot}

BASELINE (do NOT copy — extend it, max 20 lines, no comments):
```python
{baseline_code}
```
{prior_results}

TASK: Write a new `_get_rewards` method for the cartpole domain. Variant: {variant}
Requirements:
- Signature: def _get_rewards(self) -> torch.Tensor
- Hardcode the 5 scale floats at the top of the body
- Extract pole_pos, pole_vel, cart_pos, cart_vel via joint_pos/joint_vel slicing
- All state tensors are 1-D [N] — element-wise ops only, NO dim=, NO unsqueeze
- At least 2 novel components beyond the baseline
- MUST use dict + torch.stack return pattern
- No comments. Max 20 lines inside the function body.
Output ONLY the ```python ... ``` block.
"""