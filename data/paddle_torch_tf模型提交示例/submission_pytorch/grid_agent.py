import torch
import numpy as np


class GridAgent(object):
    def __init__(self, model):

        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")
        self.model = model.to(self.device)

    def predict(self, obs):
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
        action = self.model(obs)
        action_numpy = action.cpu().detach().numpy().flatten()
        return action_numpy
