import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# 1. THE GOLDILOCKS ZONE
# 1.15m allows for bent knees and dynamic movement, but kills the agent if it dives.
env = gym.make(
    "Humanoid-v4", 
    healthy_z_range=(1.15, 2.0), 
    reset_noise_scale=0.005 
)

# 2. Checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=250000,
    save_path="./humanoid_goldilocks_logs/",
    name_prefix="humanoid_goldilocks"
)

policy_kwargs = dict(net_arch=dict(pi=[400, 300], qf=[400, 300]))

# 3. The Model
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=linear_schedule(0.0003), 
    n_steps=2048, 
    batch_size=128, 
    n_epochs=10, 
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs=policy_kwargs,
    tensorboard_log="./ppo_goldilocks_tensorboard/"
)

print("Starting 3D HUMANOID Goldilocks Run (2 Million Steps)...")
model.learn(
    total_timesteps=10000000, # Crank it to 10M
    callback=checkpoint_callback,
    progress_bar=True
)

model.save("ppo_humanoid_goldilocks_final")
env.close()