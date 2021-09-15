
import numpy as np
from utils import unvec_action

class TableAgent():

    def __init__(self, settings, this_directory_path):
        self.settings = settings
        self.v_action = np.zeros(settings.num_gen)

    def act(self, obs, reward=0.0, done=False):

        adjust_gen_p = np.array(obs.nextstep_gen_p) - np.array(obs.gen_p)
        adjust_gen_v_action_space = obs.action_space['adjust_gen_v']
        adjust_gen_v = adjust_gen_v_action_space.sample()
        # adjust_v = np.array(obs.nextstep_gen_v) - np.array(obs.gen_v)
        # adjust_v = self.v_action

        act = {"adjust_gen_p": adjust_gen_p, 
                "adjust_gen_v": adjust_gen_v}
        # action = np.concatenate([adjust_gen_p, adjust_gen_v], axis=-1).astype(np.float32)
        # act = unvec_action(action, obs.action_space)
        return act
