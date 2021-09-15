
import os, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stable_baselines3 import PPO, DDPG, TD3, SAC
from utils import vec_obs, unvec_action, NUM_GEN

class Agent():
    """
        the name should be "Agent"
        input args are "settings" and "this_directory_path"
    """

    def __init__(self, settings, this_directory_path):
        self.settings = settings
        model_path = os.path.join(this_directory_path, settings["model_name"])
        self.model = SAC.load(model_path)

    def act(self, obs, reward=0.0, done=False):
        # convert class obs to vectorized observation
        observation = vec_obs(obs) 
        # predict
        action, _ = self.model.predict(observation, deterministic=True)
        # convert vectorized action to dict-like act
        act = unvec_action(action, obs.action_space) 
        return act
