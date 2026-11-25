"""Fine-tune v30 into v33 ("summit surge"): 5M more steps aimed at getting up the ramp.

v30 reliably clears the bumps but falls about two-thirds of the way up the ramp. This
stage keeps the v30 policy and normalization stats and adds rewards for upward speed and
forward momentum on the ramp (growing as it climbs) and a bonus for reaching the landing.
It trains on the demo course (landing pad lowered to z = 0.6) and lets the torso go as
low as 0.5 m before an episode ends, which leaves room for the knee slide.

    python finetune_v33.py --steps 5_000_000
"""

import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize

from course_env import LidarObservation, make_course_env


class SummitSurgeReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_action = np.zeros(self.env.action_space.shape)

    def reset(self, **kwargs):
        self.prev_action = np.zeros(self.env.action_space.shape)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        data = self.env.unwrapped.data
        x_pos, z_pos = data.qpos[0], data.qpos[2]
        forward_vel = info.get("x_velocity", 0)
        z_vel = data.qvel[2]

        if forward_vel > 0:
            reward += forward_vel * 3.0
        if 5.0 <= x_pos < 10.0:  # on the ramp
            if z_vel > 0:
                reward += z_vel * 15.0
            progress = (x_pos - 5.0) / 5.0  # 0 at the bottom, 1 at the top
            if forward_vel > 0:
                reward += forward_vel * (5.0 + progress * 15.0)
        if x_pos >= 10.0:  # on the landing pad
            reward += 30.0
            if z_pos > 1.2:
                reward += 15.0

        reward -= np.sum(np.square(action - self.prev_action)) * 0.01
        self.prev_action = action.copy()
        return obs, reward, terminated, truncated, info


def make_env():
    env = make_course_env("demo", max_episode_steps=2000, healthy_z_range=(0.5, 3.0))
    return SummitSurgeReward(LidarObservation(env))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resume", default="models/ppo_humanoid_vision_v30.zip")
    p.add_argument("--stats", default="models/vec_normalize_v30.pkl")
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=10)
    p.add_argument("--out", default="ppo_humanoid_v33_surge")
    args = p.parse_args()

    vec_env = SubprocVecEnv([make_env for _ in range(args.n_envs)])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = VecNormalize.load(args.stats, vec_env)
    vec_env.training = True
    vec_env.norm_reward = True

    hp = {
        "learning_rate": 5e-5,
        "n_steps": 4096,
        "batch_size": 10240,
        "n_epochs": 10,
        "clip_range": 0.2,
        "ent_coef": 0.005,
        "tensorboard_log": "./runs/",
    }
    model = PPO.load(args.resume, env=vec_env, custom_objects=hp)
    checkpoints = CheckpointCallback(save_freq=250_000, save_path="./checkpoints/", name_prefix=args.out)
    model.learn(total_timesteps=args.steps, callback=[checkpoints], progress_bar=True, reset_num_timesteps=False)

    model.save(args.out)
    vec_env.save(f"{args.out}_vecnormalize.pkl")
    vec_env.close()


if __name__ == "__main__":
    main()
