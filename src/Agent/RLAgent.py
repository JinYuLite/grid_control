import numpy as np

from Agent.BaseAgent import BaseAgent
from utilize.form_action import *
from stable_baselines3 import PPO, DDPG, TD3, SAC


class RLAgent(BaseAgent):

    def __init__(self, num_gen, seed=None):
        super().__init__(num_gen)
        # self.seed = seed
        # self.v_action = np.zeros(num_gen)

    def act(self, obs, reward=0.0, done=False):
        model = SAC.load("../sac_trial")
        obs = vectorize(obs)
        action = model.predict(obs, deterministic=True)
        action = unvectorize(action[0])

        return action


def vectorize(obs):
    """
        Convert real obs to gym-like obs
    """
    new_obs = []
    # gen_p, gen_q, gen_v: 机组发电
    new_obs.extend(obs.gen_p / max(np.max(np.abs(obs.gen_p)), 1e-7) )
    new_obs.extend(obs.gen_q / max(np.max(np.abs(obs.gen_q)), 1e-7) )
    new_obs.extend(obs.gen_v / max(np.max(np.abs(obs.gen_v)), 1e-7) )

    # gen status, steps_to_recover_gen, steps_to_close_gen: 机组状态, int类型
    new_obs.extend(list(obs.gen_status))
    new_obs.extend( list(obs.steps_to_recover_gen / 100.) )
    new_obs.extend( list(obs.steps_to_close_gen / 100.) )
    # print (f"This is the gen_status: {obs.gen_status}")
    # print (f"This is the steps to recover gen: {obs.steps_to_recover_gen}")
    # print (f"This is the steps to close gen: {obs.steps_to_close_gen}")

    # curstep_renewable_gen_p_max, nextstep_newable_gen_p_max: 新能源发电
    # new_obs.extend(obs.curstep_renewable_gen_p_max)
    # new_obs.extend(obs.nextstep_renewable_gen_p_max)
    renewable_gen_p_max_diff = np.array(obs.nextstep_renewable_gen_p_max) - np.array(obs.curstep_renewable_gen_p_max)
    new_obs.extend( ( renewable_gen_p_max_diff / max(np.max(np.abs(renewable_gen_p_max_diff)), 1e-7) ).tolist() )

    # load_p, load_q, load_v, nextstep_load_p: 负荷耗电
    # new_obs.extend(obs.load_p)
    new_obs.extend( obs.load_q / max(np.max(np.abs(obs.load_q)), 1e-7) )
    new_obs.extend( obs.load_v / max(np.max(np.abs(obs.load_v)), 1e-7) )
    load_p_diff = np.array(obs.nextstep_load_p) - np.array(obs.load_p)
    new_obs.extend( ( load_p_diff / max(np.max(np.abs(load_p_diff)), 1e-7) ).tolist() )
    # new_obs.extend(obs.nextstep_load_p)

    # 线路信息, 节点信息

    # M: 各个特征是否需要归一化
    observation = np.array(new_obs)
    return observation

def unvectorize(action):
    """
        Convert gym-like act to real act
    """
    action = np.array(action).flatten().astype(np.float32)
    act_dim = action.shape[0] // 2
    act = {"adjust_gen_p": np.clip(action[:act_dim], -0.05, 0.05),
           "adjust_gen_v": np.clip(action[act_dim:], 0, 1)}
    return act