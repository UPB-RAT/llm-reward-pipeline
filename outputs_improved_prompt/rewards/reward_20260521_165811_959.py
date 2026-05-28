def _get_rewards(self) -> torch.Tensor:
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    height_reward_scale = -0.02
    orientation_reward_scale = -0.02

    lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
    ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
    distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
    height = self._robot.data.root_pos_w[:, 2]
    orientation = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

    lin_vel_reward = lin_vel * lin_vel_reward_scale * self.step_dt
    ang_vel_reward = ang_vel * ang_vel_reward_scale * self.step_dt
    distance_to_goal_reward = (1 - torch.tanh(distance_to_goal / 0.8)) * distance_to_goal_reward_scale * self.step_dt
    height_reward = height * height_reward_scale * self.step_dt
    orientation_reward = orientation * orientation_reward_scale * self.step_dt

    rewards = {
        "lin_vel": lin_vel_reward,
        "ang_vel": ang_vel_reward,
        "distance_to_goal": distance_to_goal_reward,
        "height": height_reward,
        "orientation": orientation_reward,
    }

    return torch.sum(torch.stack(list(rewards.values())), dim=0)