import numpy as np

from utilize.exceptions.action_illegal_exceptions import *


def check_gen_p(adjust_gen_p, action_space_gen_p, gen_ids, eps):
    illegal_gen_ids = [i for i in gen_ids if adjust_gen_p[i] < action_space_gen_p.low[i] - eps or adjust_gen_p[i] > action_space_gen_p.high[i] + eps]
    return illegal_gen_ids

def check_gen_v(adjust_gen_v, action_space_gen_v, gen_ids, eps):
    illegal_gen_ids = [i for i in gen_ids if adjust_gen_v[i] < action_space_gen_v.low[i] - eps or adjust_gen_v[i] > action_space_gen_v.high[i] + eps]
    return illegal_gen_ids



def is_legal(act, last_obs, settings):
    """
    Returns:
        illegal_reasons(list): reasons why the action is illegal 
    """
    gen_ids = list(range(settings.num_gen))
    illegal_reasons = []

    action_space_gen_p = last_obs.action_space['adjust_gen_p']
    action_space_gen_v = last_obs.action_space['adjust_gen_v']

    adjust_gen_p = act['adjust_gen_p']
    adjust_gen_v = act['adjust_gen_v']

    gen_p_illegal_ids = check_gen_p(adjust_gen_p, action_space_gen_p, gen_ids, settings.action_allow_precision)
    gen_v_illegal_ids = check_gen_v(adjust_gen_v, action_space_gen_v, gen_ids, settings.action_allow_precision)

    if gen_p_illegal_ids:
        illegal_reasons.append(GenPOutOfActionSpace(gen_p_illegal_ids, action_space_gen_p, adjust_gen_p))
    if gen_v_illegal_ids:
        illegal_reasons.append(GenVOutOfActionSpace(gen_v_illegal_ids, action_space_gen_v, adjust_gen_v))

    return illegal_reasons == [], illegal_reasons
