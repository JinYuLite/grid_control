
import gym 
import numpy as np
import copy

from gym_utils import observation_space_inst, OBSERVATION_DIM
from gym_utils import action_space_inst, ACTION_DIM

class GridEnv(gym.Env):  
    """Custom Environment that follows gym interface"""  
    metadata = {'render.modes': ['human']}  

    def __init__(self, env_inst):

        super(GridEnv, self).__init__()    

        # this is an instance of grid environment
        self.env = env_inst

        # Define action space    
        self.action_space = action_space_inst

        # Define observation space    
        self.observation_space = observation_space_inst

        print("Action Dim: {}\nObservation Dim: {}".format(ACTION_DIM, OBSERVATION_DIM))


    def step(self, action):    
        # Convert np.array action to dict act
        act = self.action_space.from_gym(action, self.legal_act_space)
        # Run one env step 
        obs, reward, done, info = self.env.step(act)
        # Convert class obs to np.array observation
        observation = self.observation_space.to_gym(obs)
        # Save legal act space for this step
        self.legal_act_space = obs.action_space
        return observation, reward, done, info  

    def reset(self):
        obs = self.env.reset()
        observation = self.observation_space.to_gym(obs)
        # observation = vec_obs(obs)
        # Save legal act space for this step
        self.legal_act_space = obs.action_space
        return observation

    def render(self, mode="human"):
        return 

    def close(self):
        return

