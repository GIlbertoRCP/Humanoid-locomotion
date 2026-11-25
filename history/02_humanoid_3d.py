import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# 1. Setup the 3D Humanoid environment
env = gym.make("Humanoid-v4")

# 2. Checkpoint Callback: Saves every 500k steps (crucial for long runs)
checkpoint_callback = CheckpointCallback(
    save_freq=500000,
    save_path="./humanoid_logs/",
    name_prefix="humanoid_3d_model"
)

# 3. High-Parameter PPO for 3D Locomotion
# We increase the net_arch to give the AI more "neurons" for 3D balance
policy_kwargs = dict(net_arch=dict(pi=[400, 300], qf=[400, 300]))

model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=3e-4, 
    n_steps=2048, 
    batch_size=64, 
    n_epochs=10, 
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01, # Encourage more exploration in 3D space
    tensorboard_log="./ppo_humanoid_logs/",
    # policy_kwargs=policy_kwargs # Uncomment this if you want a deeper brain
)

print("Starting 3D HUMANOID Training (10 Million Steps)...")
print("This is a long-haul run. Monitor progress at http://localhost:6006")

# 4. The 10M Step Marathon
model.learn(
    total_timesteps=10000000, 
    callback=checkpoint_callback,
    progress_bar=True
)

# 5. Final Save
model.save("ppo_humanoid_final")
env.close()