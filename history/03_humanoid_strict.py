import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# 1. CLOSING THE LOOPHOLE: The Strict Environment
# healthy_z_range: The torso must stay between 1.3m and 2.0m. (Default was 1.0m)
# reset_noise_scale: We lower the starting randomization so it starts standing perfectly straight.
env = gym.make(
    "Humanoid-v4", 
    healthy_z_range=(1.15, 2.0), 
    reset_noise_scale=0.005 
)

checkpoint_callback = CheckpointCallback(
    save_freq=250000,
    save_path="./humanoid_strict_logs/",
    name_prefix="humanoid_strict_model"
)

policy_kwargs = dict(net_arch=dict(pi=[400, 300], qf=[400, 300]))

# 2. The Strict Model
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
    tensorboard_log="./ppo_humanoid_strict/"
)

print("Starting STRICT 3D HUMANOID Training (2 Million Steps)...")
print("Expect terrible scores at first. It is learning a harder task.")

model.learn(
    total_timesteps=1000000, 
    callback=checkpoint_callback,
    progress_bar=True
)

model.save("ppo_humanoid_strict_final")
env.close()