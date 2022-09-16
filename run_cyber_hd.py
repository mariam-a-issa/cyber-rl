import gym
import copy
import numpy as np
import torch
from time import time
# from QHDModel import QHDModel
import matplotlib.pyplot as plt

# From gym-idsgame
import gym
from gym_idsgame.envs.idsgame_env import *

# Set up environment
env_name = "idsgame-minimal_defense-v0"
env = gym.make(env_name)
env = env.unwrapped

# Set hyperparameters -- everything below is Yang's Code
dimension = 10000
ts = 5
tau = 1 #update every 1 episode
tau_step = 10 #update every 10 steps
epsilon = 0.2
epsilon_decay = 0.99
reward_decay = 0.9
EPISODES = 201
minimum_epsilon = 0.01

# Set up RL Model
n_actions = env.action_space.n
n_obs = env.observation_space.shape[0]
model = QHDModel(
    dimension,
    n_actions,
    n_obs,
    epsilon,
    epsilon_decay,
    minimum_epsilon,
    reward_decay,
    train_sample_size=ts,
    lr=0.05)

# Set up metrics
total_runtime = 0
total_step = 0
rewards = []

# step(), reset(), render(), and close() can be used like any other gym
for episode in range(EPISODES):
    start = time()
    rewards_sum = 0
    obs = env.reset()
    model.n_steps = 0

    while True:
        action = int(model.act(obs))
        new_obs, reward, done, info = env.step(action)

        model.store_transition(obs, action, reward, new_obs, done)
        rewards_sum += reward
        rewards.append(rewards_sum)
        model.feedback()

        total_step += 1
        if total_step % tau_step == 0:
            model.delay_model = copy.deepcopy(model.model)

        if rewards_sum > 1000:
            done = True

        if done:
            end = time()
            total_runtime += end - start
            print('Episode: ', episode)
            print('Episode Rewards: ', rewards_sum)
            break

        model.n_steps += 1
        obs = new_obs

    # Epsilon Decay
    model.epsilon = max(model.epsilon * model.epsilon_decay, model.minimum_epsilon)
