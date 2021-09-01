
import gym 
import numpy as np
import copy

from utils import vec_obs, devec_action, mask_act, NUM_GEN

class GridEnv(gym.Env):  
    """Custom Environment that follows gym interface"""  
    metadata = {'render.modes': ['human']}  

    def __init__(self, env_inst):

        super(GridEnv, self).__init__()    

        # this is an instance of grid environment
        self.env = env_inst

        # Define action space    
        ACTION_DIM = NUM_GEN * 2
        # self.action_space = gym.spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, shape=(ACTION_DIM,), dtype=np.float32) 
        action_v_low = [0] * NUM_GEN
        action_v_high = [1] * NUM_GEN 
        action_p_low = [-0.05] * NUM_GEN 
        action_p_high = [0.05] * NUM_GEN 
        action_low = np.array(action_p_low + action_v_low)
        action_high = np.array(action_p_high + action_v_high)
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32) 

        # Define observation space    
        tmp_obs = self.env.reset()
        tmp_observation = vec_obs(tmp_obs)
        OBSERVATION_DIM = tmp_observation.shape[0]
        self.observation_space = gym.spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, shape=(OBSERVATION_DIM,), dtype=np.float32) 

        print("Action Dim: {}\nObservation Dim :{}".format(ACTION_DIM, OBSERVATION_DIM))

            
    def step(self, action):    
        act = devec_action(action)
        act = mask_act(act, self.legal_act_space)

        obs, reward, done, info = self.env.step(act)
        observation = vec_obs(obs)

        # save legal act space for this step
        self.legal_act_space = obs.action_space
        return observation, reward, done, info  

    def reset(self):
        obs = self.env.reset()
        observation = vec_obs(obs)

        # save legal act space for this step
        self.legal_act_space = obs.action_space
        return observation

    def render(self, mode="human"):
        return 

    def close(self):
        return

