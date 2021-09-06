
import gym 
import numpy as np
import copy

from utils import vec_obs, unvec_action, NUM_GEN

class GridEnv(gym.Env):  
    """Custom Environment that follows gym interface"""  
    metadata = {'render.modes': ['human']}  

    def __init__(self, env_inst):

        super(GridEnv, self).__init__()    

        # this is an instance of grid environment
        self.env = env_inst

        # Define action space    
        ACTION_DIM = NUM_GEN * 2
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(ACTION_DIM,), dtype=np.float32) # normalize in step

        # Define observation space    
        tmp_obs = self.env.reset()
        tmp_observation = vec_obs(tmp_obs)
        OBSERVATION_DIM = tmp_observation.shape[0]
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(OBSERVATION_DIM,), dtype=np.float32) # normalize in step and reset
        print("Action Dim: {}\nObservation Dim: {}".format(ACTION_DIM, OBSERVATION_DIM))


    def step(self, action):    
        # Convert np.array action to dict act
        act = unvec_action(action, self.legal_act_space)
        # Run one env step 
        obs, reward, done, info = self.env.step(act)
        # Convert class obs to np.array observation
        observation = vec_obs(obs)
        # Save legal act space for this step
        self.legal_act_space = obs.action_space
        return observation, reward, done, info  

    def reset(self):
        obs = self.env.reset()
        observation = vec_obs(obs)
        # Save legal act space for this step
        self.legal_act_space = obs.action_space
        return observation

    def render(self, mode="human"):
        return 

    def close(self):
        return

