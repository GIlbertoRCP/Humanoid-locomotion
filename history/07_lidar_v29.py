import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

class VisionHikerWrapperV29(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.original_torso_mass = self.env.unwrapped.model.body_mass[1]
        self.prev_action = np.zeros(self.env.action_space.shape)
        
        # Keep the breadcrumbs to lead it up the track
        self.milestones = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
        self.milestone_idx = 0
        
        obs_shape = self.env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape + 5,), dtype=np.float32
        )

    def _get_obs_with_vision(self, obs):
        lasers = self.env.unwrapped.data.sensordata.copy()
        lasers[lasers < 0] = 10.0 
        return np.concatenate([obs, lasers]).astype(np.float32)

    def reset(self, **kwargs):
        self.prev_action = np.zeros(self.env.action_space.shape)
        self.milestone_idx = 0 
        
        obs, info = self.env.reset(**kwargs)
        m = self.env.unwrapped.model
        m.body_mass[1] = self.original_torso_mass * np.random.uniform(0.8, 1.2)
        m.geom_friction[0] = [np.random.uniform(0.7, 1.3), 0.005, 0.0001]
        return self._get_obs_with_vision(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        qpos = self.env.unwrapped.data.qpos
        qvel = self.env.unwrapped.data.qvel
        
        x_pos, y_pos, z_pos = qpos[0], qpos[1], qpos[2]
        forward_vel = info.get('x_velocity', 0)
        
        # Extract aggressive lateral/spin metrics
        lateral_vel = qvel[1] 
        yaw_vel = qvel[5] # Z-axis rotation (spinning)
        
        right_foot_x = self.env.unwrapped.data.geom_xpos[5][0]
        right_foot_z = self.env.unwrapped.data.geom_xpos[5][2] 
        left_foot_x = self.env.unwrapped.data.geom_xpos[8][0]
        left_foot_z = self.env.unwrapped.data.geom_xpos[8][2]  

        # 1. THE FEAR OF DEATH (Keep it on the track)
        if np.abs(y_pos) > 1.0:
            terminated = True
            reward -= 100.0 
            return self._get_obs_with_vision(obs), reward, terminated, truncated, info

        if terminated and not truncated:
            reward -= 300.0 # Slightly reduced so it isn't entirely terrified to experiment
            return self._get_obs_with_vision(obs), reward, terminated, truncated, info

        reward += 1.0 # Heartbeat

        # ==========================================
        # 2. AGGRESSIVE BLINDERS (THE CRAB-WALK CURE)
        # ==========================================
        # Punish drifting off center
        reward -= np.abs(y_pos) * 10.0
        # Punish strafing sideways
        reward -= np.abs(lateral_vel) * 5.0
        # Punish twisting/spinning the torso
        reward -= np.abs(yaw_vel) * 10.0

        # 3. BREADCRUMBS
        if self.milestone_idx < len(self.milestones) and x_pos > self.milestones[self.milestone_idx]:
            reward += 150.0 
            self.milestone_idx += 1

        # 4. TERRAIN ADAPTATION
        lasers = self.env.unwrapped.data.sensordata
        valid_hits = [d for d in lasers if d > 0]
        closest_obstacle = min(valid_hits) if valid_hits else 10.0

        if x_pos > 5.0:
            # --- THE MOUNTAIN ---
            if forward_vel > 0: reward += forward_vel * 5.0
            reward += z_pos * 15.0 # Aggressive pull up the hill
            energy_penalty = np.sum(np.square(action)) * 0.001
        else:
            # --- THE SPRINT & HURDLE ---
            if forward_vel < 0.5: reward -= 3.0 
            else: reward += forward_vel * 5.0 

            if closest_obstacle < 1.0:
                # THE SPLIT-HURDLE: Fixes the bunny-hop
                lead_foot = max(right_foot_z, left_foot_z)
                trail_foot = min(right_foot_z, left_foot_z)
                
                # Lead leg must go high, trail leg just needs to break contact with floor
                if lead_foot > 0.15 and trail_foot > 0.02:
                    reward += 40.0 + (forward_vel * 10.0) # Insane payout for a true sprint-hurdle
                
                energy_penalty = 0.0 
            else:
                energy_penalty = np.sum(np.square(action)) * 0.005

        # Ensure chest stays over/ahead of feet
        feet_x_center = (right_foot_x + left_foot_x) / 2.0
        com_offset = x_pos - feet_x_center
        if com_offset < -0.05:
            reward -= 10.0

        action_jerk = np.sum(np.square(action - self.prev_action))
        reward -= action_jerk * 0.01 
        self.prev_action = action.copy()
        
        reward -= energy_penalty
            
        return self._get_obs_with_vision(obs), reward, terminated, truncated, info

class CurriculumCallback(BaseCallback):
    def _on_step(self) -> bool:
        if self.num_timesteps % 1000000 == 0:
            self.model.ent_coef = max(self.model.ent_coef * 0.9, 0.001)
        return True

if __name__ == '__main__':
    print("Initializing Phase 54: The Aggressive Blinders...")

    env_kwargs = {"healthy_z_range": (0.9, 3.0), "max_episode_steps": 3000}
    
    vec_env = make_vec_env(
        "Humanoid-v4", n_envs=10, env_kwargs=env_kwargs, 
        wrapper_class=VisionHikerWrapperV29, vec_env_cls=SubprocVecEnv
    )
    vec_env = VecFrameStack(vec_env, n_stack=4)

    # Load glasses from V28
    vec_env = VecNormalize.load("vec_normalize_stats_vision_v28.pkl", vec_env)
    vec_env.training = True 

    hp = {
        "learning_rate": 0.00005, # Higher LR to violently un-train the crab-walk
        "n_steps": 4096,
        "batch_size": 10240,
        "n_epochs": 10,           
        "clip_range": 0.2,       
        "ent_coef": 0.015, # Spike exploration so it discovers the new hurdle logic
        "tensorboard_log": "./ppo_vision_hiker_v29/"
    }

    # Load V28 Brain
    model_path = "ppo_humanoid_vision_v28_final.zip" 
    print(f"Resuming from {model_path} to crush the crab-walk...")
    model = PPO.load(model_path, env=vec_env, custom_objects=hp)

    checkpoint_callback = CheckpointCallback(save_freq=250000, save_path="./vision_v29_logs/", name_prefix="humanoid_vision_v29")

    print("Starting 10M Step Discipline Run...")
    model.learn(total_timesteps=10000000, callback=[checkpoint_callback, CurriculumCallback()], progress_bar=True, reset_num_timesteps=False)
    
    model.save("ppo_humanoid_vision_v29_final")
    vec_env.save("vec_normalize_stats_vision_v29.pkl")
    vec_env.close()