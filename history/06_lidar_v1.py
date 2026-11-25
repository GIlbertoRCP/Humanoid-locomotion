import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

class VisionHikerWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.original_torso_mass = self.env.unwrapped.model.body_mass[1]
        
        # CYBERNETICS: Expand the brain's Observation Space by 5 to accept the laser data
        obs_shape = self.env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape + 5,), dtype=np.float32
        )

    def _get_obs_with_vision(self, obs):
        # Fetch the laser bounce distances from MuJoCo
        lasers = self.env.unwrapped.data.sensordata.copy()
        # If a laser shoots into the sky (returns -1), tell the brain the sky is 10 meters away
        lasers[lasers < 0] = 10.0 
        return np.concatenate([obs, lasers]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Domain Randomization (The Rugged Protocol)
        m = self.env.unwrapped.model
        m.body_mass[1] = self.original_torso_mass * np.random.uniform(0.8, 1.2)
        m.geom_friction[0] = [np.random.uniform(0.7, 1.3), 0.005, 0.0001]
        
        return self._get_obs_with_vision(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        qpos = self.env.unwrapped.data.qpos
        x_pos = qpos[0]
        z_pos = qpos[2]
        forward_vel = info.get('x_velocity', 0)

        # 1. DYNAMIC VELOCITY: Allow it to slow down when it sees/reaches the ramp!
        target_vel = 1.0 if x_pos < 5.0 else 0.5 
        reward += np.exp(-1.5 * (forward_vel - target_vel)**2) * 2.0
        
        # 2. MOUNTAIN GOAT INCENTIVE: Massive reward for climbing
        if x_pos > 5.0 and z_pos > 1.3:
            reward += (z_pos - 1.3) * 5.0 

        # 3. Symmetry Penalty (Prevents the stuck arm)
        arm_asymmetry = np.abs(qpos[17] + qpos[21])
        reward -= arm_asymmetry * 0.1 

        # 4. Energy Efficiency
        reward -= np.sum(np.square(action)) * 0.007
        
        # 5. Posture Incentive
        reward += 0.2 if z_pos > 1.3 else -0.1
            
        return self._get_obs_with_vision(obs), reward, terminated, truncated, info

class CurriculumCallback(BaseCallback):
    def _on_step(self) -> bool:
        # Slowly reduce exploration as it masters the map
        if self.num_timesteps % 1000000 == 0:
            new_ent = max(self.model.ent_coef * 0.9, 0.001)
            self.model.ent_coef = new_ent
        return True


if __name__ == '__main__':
    print("Initializing the LIDAR Vision Protocol...")

    env_kwargs = {"healthy_z_range": (0.9, 3.0)}
    
    vec_env = make_vec_env(
        "Humanoid-v4", 
        n_envs=10, 
        env_kwargs=env_kwargs, 
        wrapper_class=VisionHikerWrapper,
        vec_env_cls=SubprocVecEnv
    )
    
    # Short-Term Memory (4 Frames)
    vec_env = VecFrameStack(vec_env, n_stack=4)

    # ABSOLUTE FRESH START: Create brand new normalization glasses
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    vec_env.training = True 

    # Slightly higher learning rate since we are starting from scratch
    hp = {
        "learning_rate": 0.00003,
        "n_steps": 4096,
        "batch_size": 10240,
        "n_epochs": 10,           
        "gamma": 0.999,           
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,         
        "max_grad_norm": 0.5,
        "tensorboard_log": "./ppo_vision_hiker/"
    }

    print("Growing a brand new Vision-Enabled Brain...")
    model = PPO("MlpPolicy", vec_env, verbose=1, **hp)

    checkpoint_callback = CheckpointCallback(
        save_freq=250000, 
        save_path="./vision_logs/",
        name_prefix="humanoid_vision"
    )

    print("Starting 30M Step Vision Run. See you at the top of the ramp.")
    model.learn(
        total_timesteps=30000000,
        callback=[checkpoint_callback, CurriculumCallback()],
        progress_bar=True
    )

    model.save("ppo_humanoid_vision_final")
    vec_env.save("vec_normalize_stats_vision.pkl")
    vec_env.close()