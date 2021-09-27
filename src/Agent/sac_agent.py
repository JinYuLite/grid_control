
import os, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stable_baselines3 import PPO, DDPG, TD3, SAC
from gym_utils import observation_space_inst, OBSERVATION_DIM
from gym_utils import action_space_inst, ACTION_DIM

class Agent():
    """
        the name should be "Agent"
        input args are "settings" and "this_directory_path"
    """

    def __init__(self, settings, this_directory_path):
        self.settings = settings
        model_path = os.path.join(this_directory_path, "model")
        self.model = SAC.load(model_path)

        # Define action space    
        self.action_space = action_space_inst
        # Define observation space    
        self.observation_space = observation_space_inst

    def act(self, obs, reward=0.0, done=False):
        # convert class obs to vectorized observation
        observation = self.observation_space.to_gym(obs) 
        # predict
        action, _ = self.model.predict(observation, deterministic=True)
        # convert vectorized action to dict-like act
        act = self.action_space.from_gym(action, obs.action_space) 
        return act
