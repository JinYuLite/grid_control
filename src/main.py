# -*- coding: UTF-8 -*-
import numpy as np

from Agent.DoNothingAgent import DoNothingAgent
from Agent.RandomAgent import RandomAgent
from Environment.base_env import Environment
from utilize.settings import settings

from env_wrapper import GridEnv

def run_task(my_agent):

    env_inst = Environment(settings, "EPRIReward")
    env = GridEnv(env_inst) 

    for episode in range(max_episode):
        print('\n------ episode ', episode)
        print('------ reset ')
        obs = env.reset()

        reward = 0.0
        done = False
        
        # while not done:
        for timestep in range(max_timestep):
            print('------ step ', timestep)

            # for raw environment
            # ids = [i for i,x in enumerate(obs.rho) if x > 1.0]
            # print("gen p: ", obs.gen_p[:10])
            # print("overflow rho: ", [obs.rho[i] for i in ids])    

            # action = my_agent.act(obs, reward, done)
            # print("adjust_gen_p: ", action['adjust_gen_p'][:10])

            # for wrapped environment
            print(obs.shape)
            action = np.random.random(108).astype(np.float32)
            print(action.shape)

            obs, reward, done, info = env.step(action)
            print('info:', info)
            if done:
                break

if __name__ == "__main__":
    max_timestep = 10  # 最大时间步数
    max_episode = 2  # 回合数

    my_agent = RandomAgent(settings.num_gen)

    run_task(my_agent)
