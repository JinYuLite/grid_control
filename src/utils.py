
import copy
import numpy as np

NUM_GEN = 54

# M: Notice the normalize function in Baselines. We should carefully set low and high value
# https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/input.py


################################# 9.5 ##############################
"""
### 向量化obs: 
    - gen_p: 54, [-1,1] // 归一化除以该时刻最大值，不是time independent
    - gen_q: 54, [-1,1] // 归一化除以该时刻最大值
    - gen_v: 54, [0,1] // 归一化除以该时刻最大值
    - gen_status: 54, {0,1}
    - steps_to_recover_gen: 54, [0,0.4] // 归一化处以100
    - steps_to_close_gen: 54, [0,0.4] // 归一化处以100
    - renewable_gen_p_max_diff: 54, [-1,1] // 归一化除以该时刻最大值
    - load_p_diff: 91, [-1,1] // 归一化除以该时刻最大值，不是time independent
    - load_q: 91, [-1,1] // 归一化除以该时刻最大值
    - load_v: 91, [0,1] // 归一化除以该时刻最大值
"""
def vec_obs(obs):
    """
        Convert obs to array-like observation, range [-1,1]
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
    # new_obs.extend(obs.nextstep_load_p)
    new_obs.extend( obs.load_q / max(np.max(np.abs(obs.load_q)), 1e-7) )
    new_obs.extend( obs.load_v / max(np.max(np.abs(obs.load_v)), 1e-7) )
    load_p_diff = np.array(obs.nextstep_load_p) - np.array(obs.load_p)
    new_obs.extend( ( load_p_diff / max(np.max(np.abs(load_p_diff)), 1e-7) ).tolist() )

    observation = np.array(new_obs).astype(np.float32)
    return observation

################################# 9.5 ##############################

def unvec_action(action, legal_act_space):
    """
        Convert array-like action (range [-1,1]) to act
    """
    action = np.array(action).flatten().astype(np.float32)
    act_dim = action.shape[0] // 2
    act = {"adjust_gen_p": clip_act(action[:act_dim]*0.05, legal_act_space["adjust_gen_p"]),
           "adjust_gen_v": clip_act(action[act_dim:], legal_act_space["adjust_gen_v"])}
    return act

def clip_act(act, legal_act_space):
    return np.clip(act, legal_act_space.low, legal_act_space.high)

