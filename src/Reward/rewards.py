import math

def line_over_flow_reward(obs, settings):
    r = 1 - sum([min(i, 1) for i in obs.rho])/settings.num_line
    return r

def renewable_consumption_reward(obs, settings):
    all_gen_p = 0.0
    all_gen_p_max = 0.0
    for i, j in enumerate(settings.renewable_ids):
        all_gen_p += obs.gen_p[j]
        all_gen_p_max += obs.curstep_renewable_gen_p_max[i]
    r = all_gen_p / all_gen_p_max
    return r

def balanced_gen_reward(obs, settings):
    r = 0.0
    idx = settings.balanced_id
    if obs.gen_p[idx] > settings.max_gen_p[idx]:
        r += 1 - obs.gen_p[idx] / settings.max_gen_p[idx]
    if obs.gen_p[idx] < settings.min_gen_p[idx]:
        r += obs.gen_p[idx] / settings.min_gen_p[idx] - 1
    r = 10 * r # Ensure the range of r is [-1,1]
    return r

def running_cost_reward(obs, last_obs, settings):
    r = 0.0
    for i, name in enumerate(settings.gen_name_list):
        idx = obs.unnameindex[name]
        r -= settings.second_order_cost[i] * (obs.gen_p[idx]) ** 2 + \
            settings.first_order_cost[i] * \
            obs.gen_p[idx] + settings.constant_cost[i]
        if obs.gen_status[idx] != last_obs.gen_status[idx] and idx in settings.thermal_ids:
            r -= settings.startup_cost[i]
    r = math.exp(r) - 1
    return r

def gen_reactive_power_reward(obs, settings):
    r = 0.0
    for i in range(settings.num_gen):
        if obs.gen_q[i] > settings.max_gen_q[i]:
            r += (1 - obs.gen_q[i] / settings.max_gen_q[i])
        if obs.gen_q[i] < settings.min_gen_q[i]:
            r += (1 - settings.min_gen_q[i] / obs.gen_q[i])
    r = math.exp(r) - 1
    return r

def sub_voltage_reward(obs, settings):
    r = 0.0
    for i in range(settings.num_gen):
        if obs.gen_v[i] > settings.max_gen_v[i]:
            r += (1 - obs.gen_v[i] / settings.max_gen_v[i])
        if obs.gen_v[i] < settings.min_gen_v[i]:
            r += (1 - settings.min_gen_v[i] / obs.gen_v[i])
    r = math.exp(r) - 1
    return r

def EPRIReward(obs, last_obs, settings):
    r = settings.coeff_line_over_flow * line_over_flow_reward(obs, settings) + \
        settings.coeff_renewable_consumption * renewable_consumption_reward(obs, settings) + \
        settings.coeff_running_cost * running_cost_reward(obs, last_obs, settings) + \
        settings.coeff_balanced_gen * balanced_gen_reward(obs, settings) + \
        settings.coeff_gen_reactive_power * gen_reactive_power_reward(obs, settings) + \
        settings.coeff_sub_voltage * sub_voltage_reward(obs, settings)
    return r
