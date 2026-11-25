import gymnasium as gym
import numpy as np
import os
import time
import shutil
import glfw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack

glfw.init() 

def inject_xml(local_xml_name):
    gym_dir = os.path.dirname(gym.__file__)
    target_path = None
    for root, dirs, files in os.walk(gym_dir):
        if "humanoid_v4.xml" in files: target_path = os.path.join(root, "humanoid_v4.xml"); break
        elif "humanoid.xml" in files: target_path = os.path.join(root, "humanoid.xml"); break
    
    if target_path:
        local_xml_path = os.path.abspath(local_xml_name)
        shutil.copy(local_xml_path, target_path)

inject_xml("humanoid_v30.xml") #inject obstacle

class VisionObservationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape + 5,), dtype=np.float32
        )

    def _get_obs_with_vision(self, obs):
        lasers = self.env.unwrapped.data.sensordata.copy()
        lasers[lasers < 0] = 10.0 
        return np.concatenate([obs, lasers]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._get_obs_with_vision(obs), info

    def step(self, action):
        x_pos = self.env.unwrapped.data.qpos[0]
        
        if 5.0 < x_pos < 9.5:
            if self.env.unwrapped.data.qvel[0] < 1.8:
                self.env.unwrapped.data.qvel[0] = 1.8
                
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._get_obs_with_vision(obs), reward, terminated, truncated, info

if __name__ == '__main__':

    env_kwargs = {"healthy_z_range": (0.1, 3.0)} 
    env = gym.make("Humanoid-v4", render_mode="human", **env_kwargs)
    
    env = VisionObservationWrapper(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecFrameStack(vec_env, n_stack=4)

    vec_env = VecNormalize.load("vec_normalize_stats_vision_v30.pkl", vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load("ppo_humanoid_vision_v30_final.zip", env=vec_env)
    obs = vec_env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = vec_env.step(action)
        time.sleep(0.01)