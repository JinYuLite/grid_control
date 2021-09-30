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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack, SubprocVecEnv

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

    # 1. Set parameters
    n_stack = 1
    norm_obs, norm_reward = False, False 
    exp_name = "sac_genp"
    exp_dir = os.path.join("../outputs", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 2. Define env
    # env = Environment(settings, "EPRIReward")
    # env = GridEnv(env)  
    env = DummyVecEnv([make_env(exp_dir, rank=0)])
    eval_env = DummyVecEnv([make_env(exp_dir, rank=1)])
    # - Frame-stacking
    env = VecFrameStack(env, n_stack=n_stack)  
    eval_env = VecFrameStack(eval_env, n_stack=n_stack)  
    # - Normalize
    env = VecNormalize(env, norm_obs=norm_obs, norm_reward=norm_reward)
    eval_env = VecNormalize(eval_env, norm_obs=norm_obs, norm_reward=norm_reward)

    # 3. Create model
    policy_kwargs = dict(activation_fn=torch.nn.ReLU) 
    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs, 
        verbose=1,
        buffer_size=100000,
        train_freq=(1, "episode"),
        gamma=0.95,
        batch_size=128,
        learning_rate=0.0005,
        target_update_interval=100,
        gradient_steps=10,
        ent_coef='auto_0.1',
        tensorboard_log=exp_dir
    )

    # 4. Start training
    eval_callback = EvalCallback(eval_env, best_model_save_path=exp_dir,
                             log_path=exp_dir, eval_freq=1000, n_eval_episodes=10, 
                             deterministic=True, render=False)
    model.learn(total_timesteps=100000, log_interval=50, callback=eval_callback)
    model.save(os.path.join(exp_dir, "sac_agent"))


