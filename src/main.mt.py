# -*- coding: UTF-8 -*-
import os, sys
import numpy as np
from shutil import copyfile
from tqdm import tqdm
import time

from Agent.DoNothingAgent import DoNothingAgent
from Agent.RandomAgent import RandomAgent
from Agent.sac_agent import Agent as SACAgent
from Agent.sac_agent_stack import Agent as SACAgentStack
from Agent.sac_agent_norm import Agent as SACAgentNorm
from Agent.ppo_agent import Agent as PPOAgent
from Agent.table_agent import TableAgent
from Environment.base_env import Environment
from utilize.settings import settings

from concurrent.futures import ThreadPoolExecutor
import threading

max_turn = np.inf 

def run_episode(ep, my_agent):

    env = Environment(settings, "EPRIReward")
    # env = GridEnv(env_inst) 

    obs = env.reset()
    done, reward  = False, 0.0
    total_reward, total_step = 0.0, 0
    # print("episode: {} start_sample_idx:{}".format(ep, env.sample_idx))
    while not done:
        act = my_agent.act(obs, reward, done)
        obs, reward, done, info = env.step(act)
        total_reward += reward
        total_step += 1
        if total_step >= max_turn: break
    # print(info)
    # print("return: {} length:{}".format(total_reward, total_step))
    return total_reward, total_step

def run_task(my_agent, max_episode=10, num_workers=1):

    pool = ThreadPoolExecutor(max_workers=num_workers)
    res_l = []
 
    start_time = time.time()
    episode_rewards, episode_lengths = [], []
    for ep in range(max_episode):
        res_l.append(pool.submit(run_episode, ep, my_agent))

    res_l = [res.result() for res in res_l]
    episode_rewards = [r[0] for r in res_l]
    episode_lengths = [r[1] for r in res_l]

    mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
    mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

    print("=== {} Test on {} Episodes ===".format(my_agent.__class__.__name__,max_episode))
    print("episode_reward: {:.2f} +/- {:.2f}".format(mean_reward,std_reward))
    print("episode_length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))
    print("Cost {:.2f} seconds.\n".format(time.time()-start_time))
    return

if __name__ == "__main__":
    
    path = "./Agent"
    num_workers = 10 # 进程数
    max_episode = 10  # 回合数

    # # do noting agent
    # my_agent = DoNothingAgent(settings, path)
    # run_task(my_agent, max_episode=max_episode, num_process=num_process)

    # # random agent
    # my_agent = RandomAgent(settings, path)
    # run_task(my_agent, max_episode=max_episode, num_process=num_process)

    # rl agent
    # sac genp_t10 
    copyfile("../selected_models/sac_genp_t10.zip", os.path.join(path, "model.zip"))
    my_agent = SACAgent(settings, path)
    run_task(my_agent, max_episode=max_episode, num_workers=num_workers)
    raise

    # # with noise
    # copyfile("../outputs/sac_genp_noise/best_model.zip", os.path.join(path, "model.zip"))
    # my_agent = SACAgent(settings, path)
    # run_task(my_agent, max_episode=max_episode, num_workers=num_workers)

    # # sac stack
    # copyfile("../outputs/sac_genp_stack4/best_model.zip", os.path.join(path, "model.zip"))
    # my_agent = SACAgentStack(settings, path)
    # run_task(my_agent)

    # # ppo 
    # copyfile("../outputs/ppo_genp_p4/best_model.zip", os.path.join(path, "model.zip"))
    # my_agent = PPOAgent(settings, path)
    # run_task(my_agent)

    # table agent
    # my_agent = TableAgent(settings, path)
    # run_task(my_agent)




