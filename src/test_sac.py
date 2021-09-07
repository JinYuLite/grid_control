# -*- coding: UTF-8 -*-
import numpy as np

from utilize.settings import settings
from stable_baselines3 import PPO, DDPG, TD3, SAC

from Environment.base_env import Environment
from env_wrapper import GridEnv
# from stable_baselines3.common.evaluation import evaluate_policy

def run_test():

    max_episode = 10 

    # load agent parameters 
    # agent = SAC.load("../outputs/sac_agent")
    agent = SAC.load("sac_trial")

    # define test environment
    env_init = Environment(settings, "EPRIReward")
    env = GridEnv(env_init) 

    # run_test
    episode_rewards, episode_lengths = [], []
    for ep in range(max_episode):
        obs = env.reset()
        done = False
        total_reward, total_step = 0.0, 0

        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action[0])
            total_reward += reward
            total_step += 1
        episode_rewards.append(total_reward)
        episode_lengths.append(total_step)
        print(info)
        

    # episode_rewards, episode_lengths = evaluate_policy(
    #         agent,
    #         env,
    #         n_eval_episodes=max_episode,
    #         render=False,
    #         deterministic=True,
    #         return_episode_rewards=True
    # )

    mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
    mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

    print("=== Test on {} Episodes ===".format(max_episode))
    print("episode_reward: {:.2f} +/- {:.2f}".format(mean_reward,std_reward))
    print("episode_length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))


if __name__ == "__main__":
    run_test()

