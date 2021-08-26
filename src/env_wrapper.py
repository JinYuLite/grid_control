
import gym 
import numpy as np
import copy

class GridEnv(gym.Env):  
    """Custom Environment that follows gym interface"""  
    metadata = {'render.modes': ['human']}  

    def __init__(self, env_inst):

        super(GridEnv, self).__init__()    

        # this is an instance of grid environment
        self.env = env_inst

        # Define action and observation space    
        ACTION_DIM = 54 * 2
        self.action_space = gym.spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, shape=(ACTION_DIM,), dtype=np.float32) 

        tmp_obs = self.env.reset()
        tmp_observation = vectorize(tmp_obs)
        OBSERVATION_DIM = tmp_observation.shape[0]
        self.observation_space = gym.spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, shape=(OBSERVATION_DIM,), dtype=np.float32) 

        print("Action Dim: {}\nObservation Dim :{}".format(ACTION_DIM, OBSERVATION_DIM))

            
    def step(self, action):    
        act = unvectorize(action)
        act = self._mask_act(act)

        obs, reward, done, info = self.env.step(act)
        observation = vectorize(obs)

        # save legal act space for this step
        self.legal_act_space = obs.action_space
        return observation, reward, done, info  

    def reset(self):
        obs = self.env.reset()
        observation = vectorize(obs)

        # save legal act space for this step
        self.legal_act_space = obs.action_space
        return observation

    def render(self, mode="human"):
        return 

    def close(self):
        return

    def _mask_act(self, act):

        masked_act = {}
        for k, v in self.legal_act_space.items():
            v_low, v_high = v.low, v.high
            real_v = copy.deepcopy(act[k])
            real_v = np.where(real_v>v_low, real_v, v_low)
            real_v = np.where(real_v<v_high, real_v, v_high)
            masked_act[k] = real_v

        adjust_gen_p_low = self.legal_act_space["adjust_gen_p"].low
        adjust_gen_p_high = self.legal_act_space["adjust_gen_p"].low
        adjust_gen_v_low = self.legal_act_space["adjust_gen_v"].low
        adjust_gen_p_high = self.legal_act_space["adjust_gen_p"].low
        return masked_act


# M: Notice the normalize function in Baselines. We should carefully set low and high value
# https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/input.py
def vectorize(obs):
    """
        Convert real obs to gym-like obs
    """
    new_obs = []
    # gen_p, gen_q, gen_v: 机组发电
    new_obs.extend(obs.gen_p)
    new_obs.extend(obs.gen_q)
    new_obs.extend(obs.gen_v)

    # gen status, steps_to_recover_gen, steps_to_close_gen: 机组状态, int类型
    new_obs.extend(list(obs.gen_status))
    new_obs.extend(list(obs.steps_to_recover_gen))
    new_obs.extend(list(obs.steps_to_close_gen))

    # curstep_renewable_gen_p_max, nextstep_newable_gen_p_max: 新能源发电
    new_obs.extend(obs.curstep_renewable_gen_p_max)
    new_obs.extend(obs.nextstep_renewable_gen_p_max)

    # load_p, load_q, load_v, nextstep_load_p: 负荷耗电
    new_obs.extend(obs.load_p)
    new_obs.extend(obs.load_q)
    new_obs.extend(obs.load_v)
    new_obs.extend(obs.nextstep_load_p)

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
    act = {"adjust_gen_p": action[:act_dim],
           "adjust_gen_v": action[act_dim:]}
    return act

