"""Train (or continue training) the course-walking humanoid with PPO.

This is the v30 stage of the project. v30 was not trained from scratch: it resumed
from v29, which resumed from v28, and so on back through earlier reward versions
(see history/). Together that chain adds up to roughly 160M environment steps.

    python train.py --resume models/ppo_humanoid_vision_v30.zip \
                    --stats models/vec_normalize_v30.pkl --steps 10_000_000

The reward wrapper below is the v30 reward as it was run, including one known bug
(see the FOOT INDICES note), so that this script reproduces the published model.
"""

import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize

from course_env import LidarObservation, make_course_env

# FOOT INDICES (known bug, kept to reproduce v30): these were meant to be the right and
# left foot geoms, but in the course model geom 5 is `torso1` and geom 8 is `lwaist`.
# As a result the "lead foot height" check below is always true, "foot reach" is always
# about 0, and the lean-back check compares the torso with the waist, not the feet.
RIGHT_FOOT_GEOM = 5
LEFT_FOOT_GEOM = 8


class CourseRewardV30(gym.Wrapper):
    """v30 reward: breadcrumbs along the course, a jump zone near obstacles, and
    posture/energy terms elsewhere. Also randomizes torso mass and floor friction."""

    def __init__(self, env):
        super().__init__(env)
        self.original_torso_mass = self.env.unwrapped.model.body_mass[1]
        self.prev_action = np.zeros(self.env.action_space.shape)
        self.milestones = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
        self.milestone_idx = 0

    def reset(self, **kwargs):
        self.prev_action = np.zeros(self.env.action_space.shape)
        self.milestone_idx = 0
        obs, info = self.env.reset(**kwargs)
        m = self.env.unwrapped.model
        m.body_mass[1] = self.original_torso_mass * np.random.uniform(0.8, 1.2)
        m.geom_friction[0] = [np.random.uniform(0.7, 1.3), 0.005, 0.0001]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        data = self.env.unwrapped.data

        x_pos, y_pos, z_pos = data.qpos[0], data.qpos[1], data.qpos[2]
        forward_vel = info.get("x_velocity", 0)
        lateral_vel, yaw_vel = data.qvel[1], data.qvel[5]

        right_foot_x, right_foot_z = data.geom_xpos[RIGHT_FOOT_GEOM][0], data.geom_xpos[RIGHT_FOOT_GEOM][2]
        left_foot_x, left_foot_z = data.geom_xpos[LEFT_FOOT_GEOM][0], data.geom_xpos[LEFT_FOOT_GEOM][2]
        front_foot_x = max(right_foot_x, left_foot_x)
        com_offset = x_pos - (right_foot_x + left_foot_x) / 2.0

        # Leaving the track sideways ends the episode.
        if np.abs(y_pos) > 1.0:
            terminated = True
            reward -= 100.0
        if terminated and not truncated:
            reward -= 300.0
            return obs, reward, terminated, truncated, info

        reward += 1.0  # staying alive

        # Breadcrumbs: one-time bonus for each half-meter milestone passed.
        if self.milestone_idx < len(self.milestones) and x_pos > self.milestones[self.milestone_idx]:
            reward += 150.0
            self.milestone_idx += 1

        lasers = data.sensordata
        hits = [d for d in lasers if d > 0]
        closest_obstacle = min(hits) if hits else 10.0

        if closest_obstacle < 1.0 and x_pos < 5.0:
            # Jump zone near the bumps: no posture penalties, pay for lifting and reaching.
            if max(right_foot_z, left_foot_z) > 0.15:
                reward += 20.0 + forward_vel * 5.0
            foot_reach = front_foot_x - x_pos
            if foot_reach > 0.2:
                reward += foot_reach * 50.0
            energy_penalty = 0.0
        else:
            # Running and climbing: keep upright, straight and centered.
            if com_offset < -0.05:
                reward -= 10.0
            reward -= np.abs(y_pos) * 5.0
            reward -= np.abs(lateral_vel) * 2.0
            reward -= np.abs(yaw_vel) * 5.0
            if x_pos > 5.0:  # on the ramp
                if forward_vel > 0:
                    reward += forward_vel * 5.0
                reward += z_pos * 15.0
                energy_penalty = np.sum(np.square(action)) * 0.001
            else:  # flat ground
                if forward_vel < 0.5:
                    reward -= 3.0
                else:
                    reward += forward_vel * 5.0
                energy_penalty = np.sum(np.square(action)) * 0.005

        reward -= np.sum(np.square(action - self.prev_action)) * 0.01  # smoothness
        self.prev_action = action.copy()
        reward -= energy_penalty
        return obs, reward, terminated, truncated, info


class EntropyDecay(BaseCallback):
    """Every 1M steps, multiply the entropy coefficient by 0.9 (floor 0.001)."""

    def _on_step(self):
        if self.num_timesteps % 1_000_000 == 0:
            self.model.ent_coef = max(self.model.ent_coef * 0.9, 0.001)
        return True


def make_env():
    env = make_course_env("train", max_episode_steps=3000, healthy_z_range=(0.9, 3.0))
    return CourseRewardV30(LidarObservation(env))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resume", required=True, help="PPO .zip to continue from")
    p.add_argument("--stats", required=True, help="VecNormalize .pkl that goes with --resume")
    p.add_argument("--steps", type=int, default=10_000_000)
    p.add_argument("--n-envs", type=int, default=10)
    p.add_argument("--out", default="ppo_humanoid_continued")
    args = p.parse_args()

    vec_env = SubprocVecEnv([make_env for _ in range(args.n_envs)])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = VecNormalize.load(args.stats, vec_env)
    vec_env.training = True

    hp = {
        "learning_rate": 5e-5,
        "n_steps": 4096,
        "batch_size": 10240,
        "n_epochs": 10,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "tensorboard_log": "./runs/",
    }
    model = PPO.load(args.resume, env=vec_env, custom_objects=hp)
    checkpoints = CheckpointCallback(save_freq=250_000, save_path="./checkpoints/", name_prefix=args.out)
    model.learn(total_timesteps=args.steps, callback=[checkpoints, EntropyDecay()],
                progress_bar=True, reset_num_timesteps=False)

    model.save(args.out)
    vec_env.save(f"{args.out}_vecnormalize.pkl")
    vec_env.close()


if __name__ == "__main__":
    main()
