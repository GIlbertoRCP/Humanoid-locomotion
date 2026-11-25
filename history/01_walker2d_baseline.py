import gymnasium as gym
from stable_baselines3 import PPO

# 1. Create the Bipedal Walker environment
# Walker2d is a 2D biped that gets points for moving forward and staying upright.
env = gym.make("Walker2d-v4")

print("Initializing PPO Model for Bipedal Locomotion...")
# 2. Define the model
# We'll stick with MlpPolicy for this baseline.
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_walker_logs/")

print("Starting Biped Training (100k steps)...")
# 3. Train the agent
model.learn(total_timesteps=100_000)

# 4. Save the bipedal brain
model.save("ppo_walker_baseline")
print("Bipedal model saved as 'ppo_walker_baseline.zip'")

env.close()