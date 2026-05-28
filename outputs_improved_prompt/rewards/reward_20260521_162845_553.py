def _get_rewards(self) -> torch.Tensor:
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    tilt_reward_scale = -0.05
    height_reward_scale = -0.1

    lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
    ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
    distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
    tilt = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
    height = self._robot.data.root_pos_w[:, 2] - self.desired_pos_w[:, 2]

    lin_vel_scaled = lin_vel * lin_vel_reward_scale * self.step_dt
    ang_vel_scaled = ang_vel * ang_vel_reward_scale * self.step_dt
    distance_to_goal_scaled = (1 - torch.tanh(distance_to_goal / 0.8)) * distance_to_goal_reward_scale * self.step_dt
    tilt_scaled = tilt * tilt_reward_scale * self.step_dt
    height_scaled = height * height_reward_scale * self.step_dt

    rewards = {
        "lin_vel": lin_vel_scaled,
        "ang_vel": ang_vel_scaled,
        "distance_to_goal": distance_to_goal_scaled,
        "tilt": tilt_scaled,
        "height": height_scaled,
    }

    return torch.sum(torch.stack(list(rewards.values())), dim=0)