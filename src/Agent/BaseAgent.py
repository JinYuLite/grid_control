import copy
from abc import abstractmethod

class BaseAgent():
    def __init__(self, num_gen):
        self.num_gen = num_gen

    def reset(self, ons):
        pass

    @abstractmethod
    def act(self, obs, reward, done=False):
        pass
