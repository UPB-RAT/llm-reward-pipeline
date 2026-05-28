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
    rew_pole_angle = rew_scale_pole_pos * (torch.pi - torch.abs(pole_pos)) ** 2
    rew_cart_pos = rew_scale_cart_vel * torch.exp(-torch.abs(cart_pos) ** 2)
    rew_alive = rew_scale_alive * (1.0 - self.reset_terminated.float())
    rew_termination = rew_scale_terminated * self.reset_terminated.float()
    rewards = {
        "pole_angle": rew_pole_angle,
        "cart_pos": rew_cart_pos,
        "alive": rew_alive,
        "termination": rew_termination,
    }
    return torch.sum(torch.stack(list(rewards.values())), dim=0)