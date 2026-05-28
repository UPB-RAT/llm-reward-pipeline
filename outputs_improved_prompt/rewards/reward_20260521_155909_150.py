def _get_rewards(self) -> torch.Tensor:
    # 1. Define scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    height_reward_scale = -0.1
    tilt_reward_scale = -0.05
    velocity_error_reward_scale = -0.02
    
    # 2. Compute metrics
    lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
    ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
    distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
    height_error = torch.abs(self._robot.data.root_pos_w[:, 2] - self.desired_pos_w[:, 2])
    velocity_error = torch.linalg.norm(self._robot.data.root_lin_vel_b - self._robot.data.projected_gravity_b, dim=1)
    tilt = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
    
    # 3. Assemble dictionary
    rewards = {
        "lin_vel": lin_vel * lin_vel_reward_scale * self.step_dt,
        "ang_vel": ang_vel * ang_vel_reward_scale * self.step_dt,
        "distance_to_goal": (1 - torch.tanh(distance_to_goal / 0.8)) * distance_to_goal_reward_scale * self.step_dt,
        "height_error": height_error * height_reward_scale * self.step_dt,
        "velocity_error": velocity_error * velocity_error_reward_scale * self.step_dt,
        "tilt": tilt * tilt_reward_scale * self.step_dt,
    }
    
    # 4. Return
    return torch.sum(torch.stack(list(rewards.values())), dim=0)