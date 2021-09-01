
import os, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stable_baselines3 import PPO, DDPG, TD3, SAC
from utils import vec_obs, devec_action, mask_act, NUM_GEN

class Agent():

    def __init__(self, settings, this_directory_path):
        self.settings = settings
        
        model_path = os.path.join(this_directory_path, "model")
        self.model = SAC.load(model_path)
        

    def act(self, obs, reward=0.0, done=False):
        observation = vec_obs(obs) # convert class obs to vectorized observation
        action = self.model.predict(observation, deterministic=True) # predict
        action = self.clip_actions(action[0])  
        act = devec_action(action) # convert vectorized action to dict-like act
        act = mask_act(act, obs.action_space) # clip actions
        return act


    def clip_actions(self, actions):
        action_v_low = [0] * 54
        action_v_high = [1] * 54
        action_p_low = [-0.05] * 54
        action_p_high = [0.05] * 54
        action_low = np.array(action_p_low + action_v_low)
        action_high = np.array(action_p_high + action_v_high)
        return np.clip(actions, action_low, action_high)

