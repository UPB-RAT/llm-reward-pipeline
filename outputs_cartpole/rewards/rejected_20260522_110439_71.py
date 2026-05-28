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
    rew_pole_angle = rew_scale_pole_pos * torch.square(torch.atan2(pole_pos, self.cfg.max_cart_pos))
    rew_cart_pos = rew_scale_cart_pos * torch.abs(cart_pos)
    rew_combined = rew_scale_pole_pos * torch.exp(-torch.abs(pole_pos)) + rew_scale_cart_vel * torch.abs(cart_vel)
    rewards = {
        "alive": rew_scale_alive * (1.0 - self.reset_terminated.float()),
        "termination": rew_scale_terminated * self.reset_terminated.float(),
        "pole_pos": rew_pole_pos,
        "cart_pos": rew_cart_pos,
        "pole_angle": rew_pole_angle,
        "combined": rew_combined,
    }
    return torch.sum(torch.stack(list(rewards.values())), dim=0)