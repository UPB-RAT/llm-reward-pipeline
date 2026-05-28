from aerial_gym_dev.task.base_task import BaseTask
from aerial_gym_dev.sim.sim_builder import SimBuilder
import torch
import numpy as np

from aerial_gym_dev.utils.custom_math import *

from aerial_gym_dev.utils.logging import CustomLogger

import gymnasium as gym
from gym.spaces import Dict, Box

logger = CustomLogger("navigation_task")


def dict_to_class(dict):
    return type("ClassFromDict", (object,), dict)


class NavigationTask(BaseTask):
    def __init__(self, task_config):
        super().__init__(task_config)
        self.device = self.task_config.device
        # set the each of the elements of reward parameter to a torch tensor
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )
        logger.info("Building environment for navigation task.")
        logger.info(
            "Sim Name: {}, Env Name: {}, Robot Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
            )
        )

        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            args=self.task_config.args,
            device=self.device,
        )

        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        # Get the dictionary once from the environment and use it to get the observations later.
        # This is to avoid constant retuning of data back anf forth across functions as the tensors update and can be read in-place.
        self.obs_dict = self.sim_env.get_obs()
        if "curriculum_level" not in self.obs_dict.keys():
            self.curriculum_level = self.task_config.curriculum.min_level
            self.obs_dict["curriculum_level"] = self.curriculum_level
        else:
            self.curriculum_level = self.obs_dict["curriculum_level"]

        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)

        self.observation_space = Dict(
            {"observations": Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)}
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_transformation_function = (
            self.task_config.action_transformation_function
        )

        self.num_envs = self.sim_env.num_envs

        # Currently only the "observations" are sent to the actor and critic.
        # The "priviliged_obs" are not handled so far in sample-factory

        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "priviliged_obs": torch.zeros(
                (
                    self.sim_env.num_envs,
                    self.task_config.privileged_observation_space_dim,
                ),
                device=self.device,
                requires_grad=False,
            ),
            "collisions": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
        }

    def close(self):
        self.sim_env.delete_env()

    def reset(self):
        self.target_position[:, 0:3] = torch.rand_like(self.target_position) * 0.0
        self.infos = {}
        self.sim_env.reset()
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        self.target_position[:, 0:3] = (
            torch.rand_like(self.target_position[env_ids]) * 0.0
        )
        self.infos = {}
        self.sim_env.reset_idx(env_ids)
        return

    def render(self):
        return self.sim_env.render()

    def step(self, actions):
        # this uses the action, gets observations
        # calculates rewards, returns tuples
        # In this case, the episodes that are terminated need to be
        # first reset, and the first obseration of the new episode
        # needs to be returned.

        transformed_action = self.action_transformation_function(actions)
        self.sim_env.step(actions=transformed_action)

        # This step must be done since the reset is done after the reward is calculated.
        # This enables the robot to send back an updated state, and an updated observation to the RL agent after the reset.
        # This is important for the RL agent to get the correct state after the reset.
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(
            self.obs_dict
        )

        # logger.info(f"Curricluum Level: {self.curriculum_level}")

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0
        )

        # successes are are the sum of the environments which are to be truncated and have reached the target within a distance threshold
        successes = torch.sum(
            self.truncations
            * (
                torch.norm(
                    self.target_position - self.obs_dict["robot_position"], dim=1
                )
                < 1.0
            )
        )
        timeouts = torch.sum(self.truncations) - successes
        self.infos["successes"] = successes
        self.infos["timeouts"] = timeouts
        self.infos["crashes"] = torch.sum(self.terminations)

        self.sim_env.post_reward_calculation_step()

        self.infos = {}  # self.obs_dict["infos"]

        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()
        return return_tuple

    def get_return_tuple(self):
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def process_obs_for_task(self):
        self.task_obs["observations"][:, 0:3] = (
            self.target_position - self.obs_dict["robot_position"]
        )
        self.task_obs["observations"][:, 3:7] = self.obs_dict["robot_orientation"]
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

    def compute_rewards_and_crashes(self, obs_dict):
        robot_position = obs_dict["robot_position"]
        target_position = self.target_position
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        robot_orientation = obs_dict["robot_orientation"]
        target_orientation = torch.zeros_like(robot_orientation, device=self.device)
        target_orientation[:, 3] = 1.0

        pos_error_vehicle_frame = quat_apply_inverse(
            robot_vehicle_orientation, (target_position - robot_position)
        )
        return compute_reward(
            pos_error_vehicle_frame,
            obs_dict["crashes"],
            obs_dict["robot_actions"],
            obs_dict["robot_prev_actions"],  # TODO this is not a param anymore
            1.0,  # obs_dict["curriculum_level_multiplier"],
            self.task_config.reward_parameters,
        )


@torch.jit.script
def exp_func(x, gain, exp):
    return gain * torch.exp(-exp * x * x)


@torch.jit.script
def compute_reward(
    pos_error, crashes, action, prev_action, curriculum_level_multiplier, parameter_dict
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Dict[str, Tensor]) -> Tuple[Tensor, Tensor]
    dist = torch.norm(pos_error, dim=1)

    pos_reward = 2.0 / (1.0 + dist * dist)

    dist_reward = (20 - dist) / 20.0

    total_reward = (
        pos_reward
        + dist_reward  # + up_reward + action_diff_reward + absolute_action_reward
    )
    total_reward[:] = curriculum_level_multiplier * total_reward
    crashes[:] = torch.where(dist > 8.0, torch.ones_like(crashes), crashes)

    total_reward[:] = torch.where(
        crashes > 0.0, -2 * torch.ones_like(total_reward), total_reward
    )
    return total_reward, crashes