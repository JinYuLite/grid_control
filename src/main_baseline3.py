# -*- coding: UTF-8 -*-
import numpy as np

from Agent.DoNothingAgent import DoNothingAgent
from Agent.RandomAgent import RandomAgent
from Environment.base_env import Environment
from utilize.settings import settings
from stable_baselines3 import PPO, DDPG
from env_wrapper import GridEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

def run_test(my_agent):

    for episode in range(max_episode):
        print('------ episode ', episode)
        env_init = Environment(settings, "EPRIReward")
        env = GridEnv(env_init) 
        print('------ reset ')
        obs = env.reset()

        reward = 0.0
        done = False
        
        # while not done:
        for timestep in range(max_timestep):
            # ids = [i for i,x in enumerate(obs.rho) if x > 1.0]
            # print("overflow rho: ", [obs.rho[i] for i in ids])    
            print('------ step ', timestep)
            action = my_agent.predict(obs, deterministic=True)
            # print("adjust_gen_p: ", action['adjust_gen_p'])
            # print("adjust_gen_v: ", action['adjust_gen_v'])
            # print (f"THis is the action: \n{action}")
            # print ("\n")
            obs, reward, done, info = env.step(action[0])
            print('info:', info)
            print (f'This is the reward: {reward}')
            if done:
                break

if __name__ == "__main__":
    max_timestep = 10  # 最大时间步数
    max_episode = 10  # 回合数

    # my_agent = RandomAgent(settings.num_gen)

    # run_task(my_agent)
    env = Environment(settings, "EPRIReward")
    env = GridEnv(env)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)

    model.learn(total_timesteps=1000, log_interval=10)

    run_test(model)