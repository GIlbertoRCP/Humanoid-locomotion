"""Gymnasium's Humanoid-v4, loaded from this repo's course XML instead of the stock model.

Humanoid-v4 hard-codes its XML file, so the original project overwrote the file inside
the installed Gymnasium package. Here the same environment is built from a local XML,
which gives the exact same 376-value observation (plus 5 LiDAR readings) without
modifying anything outside the repo.
"""

import os

import gymnasium as gym
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco.humanoid_v4 import DEFAULT_CAMERA_CONFIG, HumanoidEnv
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.wrappers import TimeLimit

HERE = os.path.dirname(os.path.abspath(__file__))
COURSES = {
    "train": os.path.join(HERE, "assets", "course.xml"),
    "demo": os.path.join(HERE, "assets", "course_demo.xml"),
}


class CourseHumanoidEnv(HumanoidEnv):
    """Humanoid-v4 with a custom XML file. Same reward, termination and observation."""

    def __init__(
        self,
        xml_file,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self, xml_file, forward_reward_weight, ctrl_cost_weight, healthy_reward,
            terminate_when_unhealthy, healthy_z_range, reset_noise_scale, **kwargs,
        )
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = True
        MujocoEnv.__init__(
            self,
            xml_file,
            5,
            observation_space=Box(low=-np.inf, high=np.inf, shape=(376,), dtype=np.float64),
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )


class LidarObservation(gym.Wrapper):
    """Append the 5 rangefinder readings to the observation. A miss (-1) becomes 10 m."""

    def __init__(self, env):
        super().__init__(env)
        n = self.env.observation_space.shape[0] + 5
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(n,), dtype=np.float32)

    def _with_lidar(self, obs):
        lasers = self.env.unwrapped.data.sensordata.copy()
        lasers[lasers < 0] = 10.0
        return np.concatenate([obs, lasers]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._with_lidar(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._with_lidar(obs), reward, terminated, truncated, info


def make_course_env(course="train", max_episode_steps=3000, render_mode=None, **kwargs):
    """Course humanoid with a time limit, before any LiDAR or reward wrapper."""
    path = COURSES[course]
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run `python build_course.py` first.")
    env = CourseHumanoidEnv(path, render_mode=render_mode, **kwargs)
    return TimeLimit(env, max_episode_steps=max_episode_steps)
