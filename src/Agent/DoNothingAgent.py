import numpy as np

from Agent.BaseAgent import BaseAgent
from utilize.form_action import *

class DoNothingAgent(BaseAgent):

    def __init__(self, num_gen):
        BaseAgent.__init__(self, num_gen)
        self.action = form_action(np.zeros(self.num_gen), np.zeros(self.num_gen))

    def act(self, obs, reward, done=False):
        return self.action

