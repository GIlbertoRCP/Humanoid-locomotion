import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

class HikerWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        forward_vel = info.get('x_velocity', 0)
        
        target_speed = 1.0 # Slower target pace for hiking
        speed_reward = 2.0 - abs(forward_vel - target_speed)
        
        reward += speed_reward
        reward -= np.sum(np.square(action)) * 0.015 
        return obs, reward, terminated, truncated, info

print("Initializing the Obstacle Course...")

# Standard v4 (But with our injected physical floor!)
env_kwargs = {
    "healthy_z_range": (0.9, 3.0), # Massively expanded so it doesn't die walking up a ramp
    "reset_noise_scale": 0.005
}

vec_env = make_vec_env("Humanoid-v4", n_envs=8, env_kwargs=env_kwargs, wrapper_class=HikerWrapper)

try:
    vec_env = VecNormalize.load("vec_normalize_stats_pacer.pkl", vec_env)
    vec_env.training = True 
    vec_env.norm_reward = True
except FileNotFoundError:
    print("ERROR: Could not find vec_normalize_stats_pacer.pkl")
    exit()

custom_objects = {
    "learning_rate": 0.00001, # Slow and steady learning
    "n_steps": 4096,          
    "batch_size": 32768,      # MASSIVE BATCH: Look at 8 robots simultaneously to average out the trips and falls.
    "n_epochs": 5,            
    "clip_range": 0.1,        # Tight jacket to prevent brain-scrambling
    "target_kl": None,        # EXPLICITLY DISABLED: Let it learn from the speed bumps!
    "tensorboard_log": "./ppo_terrain_unbound/"
}

model_path = "ppo_humanoid_pacer_final.zip"
print(f"Loading 7-Second Champion from {model_path}...")
model = PPO.load(model_path, env=vec_env, custom_objects=custom_objects)

checkpoint_callback = CheckpointCallback(
    save_freq=125000, 
    save_path="./humanoid_terrain_logs/",
    name_prefix="humanoid_terrain"
)

print("Starting TERRAIN Training (10 Million Steps)...")
model.learn(
    total_timesteps=10000000,
    callback=checkpoint_callback,
    progress_bar=True,
    reset_num_timesteps=False 
)

model.save("ppo_humanoid_terrain_final")
vec_env.save("vec_normalize_stats_terrain.pkl")
vec_env.close()