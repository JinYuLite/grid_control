import numpy as np

from Agent.BaseAgent import BaseAgent
from utilize.form_action import *

class RandomAgent(BaseAgent):

    def __init__(self, num_gen, seed=None):
        BaseAgent.__init__(self, num_gen)
        self.seed = seed
        self.v_action = np.zeros(num_gen)

    def act(self, obs, reward=0.0, done=False):
        adjust_gen_p_action_space = obs.action_space['adjust_gen_p']
        adjust_gen_v_action_space = obs.action_space['adjust_gen_v']
        
        if self.seed is not None:
            # To make sure sample same value
            adjust_gen_p_action_space.np_random.seed(self.seed)
            adjust_gen_v_action_space.np_random.seed(self.seed)

        adjust_gen_p = adjust_gen_p_action_space.sample()
        #adjust_gen_v = adjust_gen_v_action_space.sample()
        adjust_gen_v = self.v_action
        return form_action(adjust_gen_p, adjust_gen_v)
