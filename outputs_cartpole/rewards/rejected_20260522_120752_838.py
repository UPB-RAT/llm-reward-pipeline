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
    rew_cart_pos = rew_scale_cart_pos * torch.abs(cart_pos)  # Novel component
    rew_angle_diff = rew_scale_angle_diff * torch.abs(pole_pos - torch.pi / 2)  # Novel component
    rewards = {
        "alive": rew_alive,
        "termination": rew_termination,
        "pole_pos": rew_pole_pos,
        "cart_vel": rew_cart_vel,
        "pole_vel": rew_pole_vel,
        "cart_pos": rew_cart_pos,
        "angle_diff": rew_angle_diff,
    }
    return torch.sum(torch.stack(list(rewards.values())), dim=0)