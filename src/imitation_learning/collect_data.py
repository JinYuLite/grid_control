# -*- coding: UTF-8 -*-
import os, sys
import numpy as np
from tqdm import tqdm
import torch
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from Agent.DoNothingAgent import DoNothingAgent
from Agent.RandomAgent import RandomAgent
from Agent.sac_agent import Agent as SACAgent
from Agent.table_agent import TableAgent
from Environment.base_env import Environment
from utilize.settings import settings
from utils import vec_obs, unvec_action

def run_task(my_agent, save_path, max_episode=10, max_turn=np.inf):

    env = Environment(settings, "EPRIReward")
    # env = GridEnv(env_inst) 

    episode_rewards, episode_lengths, observations, actions =  [], [], [], []
    for ep in tqdm(range(max_episode)):
        obs = env.reset()
        done, reward  = False, 0.0
        total_reward, total_step = 0.0, 0
        while not done:
            act = my_agent.act(obs, reward, done)

            # save vectorized obs/act for bc
            obs_vec = vec_obs(obs)
            def vec_act(act):
                """ gen_p归一化到[-1,1]
                """
                act_p = (act["adjust_gen_p"] - np.min(act["adjust_gen_p"])) / (np.max(act["adjust_gen_p"])-np.min(act["adjust_gen_p"])) * 2 - 1
                act_v = (act["adjust_gen_v"] - np.min(act["adjust_gen_v"])) / (np.max(act["adjust_gen_v"])-np.min(act["adjust_gen_v"])) * 2 - 1
                return np.concatenate([act_p, act_v], axis=-1)
            act_vec = vec_act(act)
            observations.append(obs_vec)
            actions.append(act_vec)

            # env step to next step 
            obs, reward, done, info = env.step(act)
            total_reward += reward
            total_step += 1
            if total_step >= max_turn: break
        episode_rewards.append(total_reward)
        episode_lengths.append(total_step)

    # write
    data = {"observations": np.stack(observations), 
            "actions": np.stack(actions)}
    torch.save(data, save_path)
    print("Saving {} samples into file {}".format(len(actions), save_path))

    mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
    mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

    print("=== {} Test on {} Episodes ===".format(my_agent.__class__.__name__,max_episode))
    print("episode_reward: {:.2f} +/- {:.2f}".format(mean_reward,std_reward))
    print("episode_length: {:.2f} +/- {:.2f}\n".format(mean_ep_length, std_ep_length))

if __name__ == "__main__":
    
    # do noting agent
    # my_agent = DoNothingAgent(settings, path)
    # run_task(my_agent)

    # # random agent
    # my_agent = RandomAgent(settings, path)
    # run_task(my_agent)

    # # rl agent
    # my_agent = SACAgent(settings, path)
    # run_task(my_agent)

    # table agent
    max_episode=int(1e4)
    save_path = "imitation_learning/table_expert.txt"
    my_agent = TableAgent(settings, "")
    run_task(my_agent, save_path, max_episode=max_episode, max_turn=1)

