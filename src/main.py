# -*- coding: UTF-8 -*-
import os, sys
import numpy as np
from shutil import copyfile
import time

from Agent.DoNothingAgent import DoNothingAgent
from Agent.RandomAgent import RandomAgent
from Agent.sac_agent import Agent as SACAgent
from Agent.table_agent import TableAgent
from Environment.base_env import Environment
from utilize.settings import settings

max_turn = 288

def run_task(my_agent):

    max_episode = 100  # 回合数

    env = Environment(settings, "EPRIReward")
    # env = GridEnv(env_inst) 
 
    start_time = time.time()
    episode_rewards, episode_lengths = [], []
    for ep in range(max_episode):
        obs = env.reset()
        done, reward  = False, 0.0
        total_reward, total_step = 0.0, 0

        while not done:
            act = my_agent.act(obs, reward, done)
            obs, reward, done, info = env.step(act)
            total_reward += reward
            total_step += 1
            if total_step >= max_turn: break
        episode_rewards.append(total_reward)
        episode_lengths.append(total_step)
        # print(info)

    mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
    mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

    print("=== {} Test on {} Episodes ===".format(my_agent.__class__.__name__,max_episode))
    print("episode_reward: {:.2f} +/- {:.2f}".format(mean_reward,std_reward))
    print("episode_length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))
    print("Cost {:.2f} seconds.\n".format(time.time()-start_time))

if __name__ == "__main__":
    
    path = "./Agent"

    # do noting agent
    my_agent = DoNothingAgent(settings, path)
    run_task(my_agent)

    # random agent
    my_agent = RandomAgent(settings, path)
    run_task(my_agent)

    # rl agent
    # sac genp_t10 
    copyfile("../selected_models/sac_genp_t10.zip", os.path.join(path, "model.zip"))
    my_agent = SACAgent(settings, path)
    run_task(my_agent)

    # with noise
    copyfile("../outputs/sac_genp_noise/best_model.zip", os.path.join(path, "model.zip"))
    my_agent = SACAgent(settings, path)
    run_task(my_agent, max_episode=max_episode, num_process=num_process)

