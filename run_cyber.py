import gym
import copy
import numpy as np
import torch
from time import time
# from QHD_QuantModel import QHD_Model
import matplotlib.pyplot as plt

# From gym-idsgame
import gym
from gym_idsgame.agents.training_agents.q_learning.q_agent_config import QAgentConfig
from gym_idsgame.agents.training_agents.q_learning.tabular_q_learning.tabular_q_agent import TabularQAgent
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig
from gym_idsgame.agents.training_agents.policy_gradient.reinforce.reinforce import ReinforceAgent

############# CODE FROM GYM-IDS

# step(), reset(), render(), and close() can be used like any other gym

# from gym_idsgame.envs import IdsGameEnv
# env_name = "idsgame-maximal_attack-v3" # TODO: can change this
# env = gym.make(env_name)
#
# random_seed = 0
# util.create_artefact_dirs(default_output_dir(), random_seed)
# q_agent_config = QAgentConfig(gamma=0.999, alpha=0.0005, epsilon=1, render=False, eval_sleep=0.9,
#                               min_epsilon=0.01, eval_episodes=100, train_log_frequency=100,
#                               epsilon_decay=0.9999, video=True, eval_log_frequency=1,
#                               video_fps=5, video_dir=default_output_dir() + "/results/videos/" + str(random_seed), num_episodes=20001,
#                               eval_render=False, gifs=True, gif_dir=default_output_dir() + "/results/gifs/" + str(random_seed),
#                               eval_frequency=1000, attacker=True, defender=False, video_frequency=101,
#                               save_dir=default_output_dir() + "/results/data/" + str(random_seed))
# env_name = "idsgame-minimal_defense-v2"
# env = gym.make(env_name, save_dir=default_output_dir() + "/results/data/" + str(random_seed))
# attacker_agent = TabularQAgent(env, q_agent_config)
# attacker_agent.train()
# train_result = attacker_agent.train_result
# eval_result = attacker_agent.eval_result

############# CODE FROM GYM-IDS

# Set Up Game Environment
random_seed = 0
default_output_dir = "./results/"
# env_name = "idsgame-maximal_attack-v3"
# env = gym.make(env_name)
# # env = gym.make(env_name, output_dir + str(random_seed))
#
#
#
# # Set up Model
# q_agent_config = QAgentConfig(
#     gamma=0.999,
#     alpha=0.0005,
#     epsilon=1,
#     render=False,
#     eval_sleep=0.9,
#     min_epsilon=0.01,
#     eval_episodes=100,
#     train_log_frequency=100,
#     epsilon_decay=0.9999,
#     video=True,
#     eval_log_frequency=1,
#     video_fps=5,
#     video_dir=output_dir + str(random_seed),
#     num_episodes=10, #20001,
#     eval_render=False,
#     gifs=True,
#     gif_dir=output_dir + str(random_seed),
#     eval_frequency=1000,
#     attacker=True,
#     defender=False,
#     video_frequency=101,
#     save_dir=output_dir + str(random_seed))
#
# attacker_agent = TabularQAgent(env, q_agent_config)
# attacker_agent.train()
# train_result = attacker_agent.train_result
# eval_result = attacker_agent.eval_result


### REINFORCE AGENT
pg_agent_config = PolicyGradientAgentConfig(gamma=0.999, alpha_attacker=0.00001, epsilon=1, render=False,
                                            eval_sleep=0.9,
                                            min_epsilon=0.01, eval_episodes=100, train_log_frequency=100,
                                            epsilon_decay=0.9999, video=True, eval_log_frequency=1,
                                            video_fps=5, video_dir="./results/videos/" + str(random_seed),
                                            num_episodes=200001,
                                            eval_render=False, gifs=True,
                                            gif_dir="./results/gifs/" + str(random_seed),
                                            eval_frequency=10000, attacker=True, defender=False,
                                            video_frequency=101,
                                            save_dir="./results/data/" + str(random_seed),
                                            checkpoint_freq=5000, input_dim_attacker=44, output_dim_attacker=40,
                                            hidden_dim=64,
                                            num_hidden_layers=1, batch_size=32,
                                            gpu=False, tensorboard=True,
                                            tensorboard_dir="./results/tensorboard/" + str(random_seed),
                                            optimizer="Adam", lr_exp_decay=False, lr_decay_rate=0.999)
env_name = "idsgame-minimal_defense-v9"
env = gym.make(env_name, save_dir="./results/data/" + str(random_seed))
attacker_agent = ReinforceAgent(env, pg_agent_config)
attacker_agent.train()
train_result = attacker_agent.train_result
eval_result = attacker_agent.eval_result

