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
from stable_baselines3.common.callbacks import EvalCallback


def make_env(save_path, rank=0):
    def _init():
        env = Environment(settings, "EPRIReward")
        env = GridEnv(env)  
        log_file = os.path.join(save_path, "env_"+str(rank)+".log")
        env = Monitor(env, log_file, allow_early_resets=True)
        return env
    return _init

if __name__ == "__main__":

    exp_name = "td3_genp_t10"
    exp_dir = os.path.join("../outputs", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Define env
    # env = Environment(settings, "EPRIReward")
    # env = GridEnv(env)  
    env = DummyVecEnv([make_env(exp_dir, rank=0)])
    eval_env = DummyVecEnv([make_env(exp_dir, rank=1)])

    # Create model
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))
    model = TD3("MlpPolicy", 
                env, 
                action_noise=action_noise, 
                verbose=1, 
                learning_rate=0.001,
                buffer_size=5000,
                learning_starts=100,
                batch_size=128,
                policy_delay=5,
                tensorboard_log=exp_dir
    )

    # Start training
    eval_callback = EvalCallback(eval_env, best_model_save_path=exp_dir,
                             log_path=exp_dir, eval_freq=1000, n_eval_episodes=10, 
                             deterministic=True, render=False)
    model.learn(total_timesteps=100000, log_interval=50, callback=eval_callback)
    model.save(os.path.join(exp_dir, "td3_agent"))


