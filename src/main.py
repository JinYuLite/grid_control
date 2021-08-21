# -*- coding: UTF-8 -*-
import numpy as np

from Agent.DoNothingAgent import DoNothingAgent
from Agent.RandomAgent import RandomAgent
from Environment.base_env import Environment
from utilize.settings import settings

def run_task(my_agent):

    for episode in range(max_episode):
        print('------ episode ', episode)
        env = Environment(settings, "EPRIReward")
        print('------ reset ')
        obs = env.reset()

        reward = 0.0
        done = False
        
        # while not done:
        for timestep in range(max_timestep):
            ids = [i for i,x in enumerate(obs.rho) if x > 1.0]
            # print("overflow rho: ", [obs.rho[i] for i in ids])    
            print('------ step ', timestep)
            action = my_agent.act(obs, reward, done)
            # print("adjust_gen_p: ", action['adjust_gen_p'])
            # print("adjust_gen_v: ", action['adjust_gen_v'])
            obs, reward, done, info = env.step(action)
            print('info:', info)
            if done:
                break

if __name__ == "__main__":
    max_timestep = 10  # 最大时间步数
    max_episode = 1  # 回合数

    my_agent = RandomAgent(settings.num_gen)

    run_task(my_agent)
