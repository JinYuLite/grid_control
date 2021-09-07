# -*- coding: UTF-8 -*-
import os, sys
import numpy as np
import torch 

from Environment.base_env import Environment
from utilize.settings import settings
from stable_baselines3 import PPO, DDPG, TD3, SAC
from env_wrapper import GridEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from stable_baselines3.common.monitor import Monitor

def make_env(save_path, rank=0):
    def _init():
        env = Environment(settings, "EPRIReward")
        env = GridEnv(env)  
        log_file = os.path.join(save_path, "env_"+str(rank)+".log")
        env = Monitor(env, log_file, allow_early_resets=True)
        return env
    return _init

if __name__ == "__main__":

    save_path = "../outputs"

    # Define env
    # env = Environment(settings, "EPRIReward")
    # env = GridEnv(env)  
    env = DummyVecEnv([make_env(save_path, rank=0)])
    # env = VecNormalize(env, norm_obs=True, norm_reward=True)

    policy_kwargs = dict(activation_fn=torch.nn.Tanh) # output: [-1,1]
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))
    # model = TD3("MlpPolicy", 
    #             env, 
    #             action_noise=action_noise, 
    #             verbose=1, 
    #             learning_rate=0.001,
    #             buffer_size=5000,
    #             learning_starts=100,
    #             batch_size=128,
    #             policy_delay=5
    #         )

    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs, 
        verbose=1,
        buffer_size=100000,
        train_freq=50,
        gamma=0.95,
        batch_size=128,
        learning_rate=0.0005,
        target_update_interval=100,
        gradient_steps=10
    )

    model.learn(total_timesteps=100000, log_interval=10)
    model.save(os.path.join(save_path, "sac_agent"))
    # env.save(os.path.join(save_path, "norm_env"))


