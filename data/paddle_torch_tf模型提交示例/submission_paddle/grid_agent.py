import paddle
import numpy as np


class GridAgent(object):
    def __init__(self, model):
        self.model = model

    def predict(self, obs):
        obs = paddle.to_tensor(obs.reshape(1, -1), dtype='float32')
        action = self.model(obs)
        action_numpy = action.cpu().numpy()[0]
        return action_numpy
