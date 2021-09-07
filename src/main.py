# -*- coding: UTF-8 -*-
import numpy as np

from Agent.DoNothingAgent import DoNothingAgent
from Agent.RandomAgent import RandomAgent
from Agent.sac_agent import Agent as SACAgent
from Agent.table_agent import TableAgent
from Environment.base_env import Environment
from utilize.settings import settings

def run_task(my_agent, max_turn=np.inf):

    max_episode = 10  # 回合数

    env = Environment(settings, "EPRIReward")
    # env = GridEnv(env_inst) 
 
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
        print(info)

    mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
    mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

    print("=== {} Test on {} Episodes ===".format(my_agent.__class__.__name__,max_episode))
    print("episode_reward: {:.2f} +/- {:.2f}".format(mean_reward,std_reward))
    print("episode_length: {:.2f} +/- {:.2f}\n".format(mean_ep_length, std_ep_length))

if __name__ == "__main__":
    
    path = "../outputs"

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
    my_agent = TableAgent(settings, path)
    run_task(my_agent, max_turn=1)




