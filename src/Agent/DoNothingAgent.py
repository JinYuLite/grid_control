import numpy as np

from utilize.form_action import *

class DoNothingAgent():

    def __init__(self, settings, file_path):
        self.num_gen = settings.num_gen
        self.action = form_action(np.zeros(self.num_gen), np.zeros(self.num_gen))

    def act(self, obs, reward, done=False):
        return self.action

