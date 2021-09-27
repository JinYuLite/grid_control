# -*- coding: UTF-8 -*-
import os, sys
import numpy as np
from tqdm import tqdm

from utilize.settings import settings
from stable_baselines3 import PPO, DDPG, TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

from Environment.base_env import Environment
from env_wrapper import GridEnv

from train_sac import make_env


def run_test():

    max_episode = 100
    exp_name = "sac_genp_t10"
    exp_dir = os.path.join("../outputs", exp_name)

    # load agent parameters 
    agent = SAC.load(os.path.join(exp_dir, "best_model"))

    # define test environment
    # env_init = Environment(settings, "EPRIReward")
    # env = GridEnv(env_init) 
    env = DummyVecEnv([make_env(exp_dir, rank=99)])

    # version 1: run_test
    episode_rewards, episode_lengths = [], []
    for ep in tqdm(range(max_episode)):
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

    # version 2: run_test with stable-baselines3
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

