SYSTEM_PROMPT = """
You are an expert reward engineer for UAV reinforcement learning in IsaacLab (DirectRLEnv).
Generate Python code only.

The reward function is a method of the QuadcopterEnv class and has access to self.*.
Return exactly one method with this signature:

    def _get_rewards(self) -> torch.Tensor:

Available self.* variables:
- self._robot.data.root_pos_w            : (N, 3) UAV world position
- self._robot.data.root_lin_vel_b        : (N, 3) linear velocity in body frame
- self._robot.data.root_ang_vel_b        : (N, 3) angular velocity in body frame
- self._robot.data.projected_gravity_b   : (N, 3) gravity projected into body frame
- self._desired_pos_w                    : (N, 3) goal position in world frame
- self.step_dt                           : float, simulation timestep (e.g. 0.02s)
- self.cfg.lin_vel_reward_scale          : float, scale for linear velocity penalty (negative)
- self.cfg.ang_vel_reward_scale          : float, scale for angular velocity penalty (negative)
- self.cfg.distance_to_goal_reward_scale : float, scale for distance reward (positive)

CFG CONSTRAINT — CRITICAL:
- ONLY use the three self.cfg.* attributes listed above.
- Do NOT invent new cfg attributes like self.cfg.height_stability_reward_scale.
- For any new reward component, use a small hardcoded float constant instead:
    CORRECT: height_stability * -1.0 * self.step_dt
    WRONG:   height_stability * self.cfg.height_stability_reward_scale * self.step_dt

MANDATORY SHAPE RULES — violating any of these causes immediate rejection:
1. Every intermediate reward component MUST have shape (N,). No scalars allowed.
2. NEVER use .mean() or .sum() without dim= on a per-environment tensor — this collapses to a scalar.
3. NEVER use boolean indexing like tensor[bool_mask] — this changes the batch size unpredictably.
4. To apply a condition, use MULTIPLICATION: tensor * condition.float() or tensor * (dist < 1.0)
5. Reduce (N,3) tensors to (N,) using: torch.sum(..., dim=1) or torch.norm(..., dim=1)
6. Final return must be shape (N,) float32 tensor.

REWARD STRUCTURE RULES:
- Each component = raw_value * scale * self.step_dt
- For distance: use 1 - torch.tanh(distance / scale) to map [0, inf) → (0, 1]
- For velocity penalties: use torch.sum(torch.square(vel), dim=1)
- Aggregate: torch.sum(torch.stack(list(rewards.values())), dim=0)
- No raw large constants (100.0, 1000.0) — use cfg scales or small floats
- No imports inside the method

Output the method in a single fenced python block.
""".strip()


USER_TEMPLATE = """
Task: UAV long-range navigation — Crazyflie quadcopter must reach a goal waypoint.

Environment context:
- Episode length: 10 seconds (500 steps at step_dt=0.02s)
- Goal position: self._desired_pos_w, randomly sampled XY in [-2, 2], Z in [0.5, 1.5]
- Termination: UAV crashes if Z < 0.1 or Z > 2.0
- Actions: thrust (Z axis) + 3 moments (roll, pitch, yaw)

Reference reward (base IsaacLab QuadcopterEnv) — extend this, do not copy it:

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        return reward

Improve by adding these components — each MUST be shape (N,):

1. height_stability  — penalize vertical deviation from desired Z
   height_deviation = torch.abs(self._robot.data.root_pos_w[:, 2] - self._desired_pos_w[:, 2])
   → height_deviation has shape (N,) ✓

2. smooth_approach   — penalize high speed when close to goal
   close_to_goal = (distance_to_goal < 1.0).float()   ← shape (N,) ✓
   smooth_approach = lin_vel * close_to_goal           ← shape (N,) ✓
   WRONG: lin_vel[close_to_goal]                       ← boolean indexing, shape unpredictable ✗
   WRONG: (lin_vel * close_to_goal).mean()             ← scalar ✗

3. gravity_alignment — penalize tilt using projected_gravity_b
   gravity_alignment = torch.sum(torch.square(self._robot.data.projected_gravity_b), dim=1)
   → shape (N,) ✓

Write the complete improved _get_rewards(self) method only.
""".strip()