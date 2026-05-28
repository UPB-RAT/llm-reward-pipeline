def _get_rewards(self) -> torch.Tensor:
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    height_reward_scale = -0.1
    orientation_reward_scale = -0.02

    lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
    ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
    distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
    height = self._robot.data.root_pos_w[:, 2]
    orientation = torch.sum(torch.abs(self._robot.data.root_ang_vel_b), dim=1)

    lin_vel_mapped = torch.tanh(lin_vel / 0.1)
    ang_vel_mapped = torch.tanh(ang_vel / 0.1)
    height_mapped = torch.tanh(height - 1.0)
    orientation_mapped = torch.tanh(orientation / 0.1)

    rewards = {
        "lin_vel": lin_vel * lin_vel_reward_scale * self.step_dt,
        "ang_vel": ang_vel * ang_vel_reward_scale * self.step_dt,
        "distance_to_goal": distance_to_goal_mapped * distance_to_goal_reward_scale * self.step_dt,
        "height": height_mapped * height_reward_scale * self.step_dt,
        "orientation": orientation_mapped * orientation_reward_scale * self.step_dt,
    }
    return torch.sum(torch.stack(list(rewards.values())), dim=0)