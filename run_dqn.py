import os
import gym
import sys
import time
from gym_idsgame.agents.training_agents.q_learning.q_agent_config import QAgentConfig
from gym_idsgame.agents.training_agents.q_learning.dqn.dqn import DQNAgent
from gym_idsgame.agents.training_agents.q_learning.dqn.dqn_config import DQNConfig
from experiments.util import util

def get_script_path():
    """
    :return: the script path
    """
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def default_output_dir() -> str:
    """
    :return: the default output dir
    """
    script_dir = get_script_path()
    return script_dir

# Program entrypoint
if __name__ == '__main__':
    scenario = str(sys.argv[1])
    attacker = True if scenario == "minimal_defense" or scenario == "random_defense" else False

    random_seed = 0
    util.create_artefact_dirs(default_output_dir(), random_seed)

    dqn_config = DQNConfig(input_dim=88,
                           defender_output_dim=88,
                           attacker_output_dim=80,
                           replay_memory_size=10000,
                           batch_size=32,
                           target_network_update_freq=1000,
                           gpu=False,
                           tensorboard=False,
                           tensorboard_dir=default_output_dir() + "./results/tensorboard/",
                           lr_exp_decay=True,
                           lr_decay_rate=0.9999,
                           num_hidden_layers=1,
                           hidden_dim=64,
                           replay_start_size=0,
                           loss_fn="Huber",
                           optimizer="Adam",
                           )
    q_agent_config = QAgentConfig(gamma=0.999,
                                  alpha=0.00001,
                                  num_episodes=20001,
                                  epsilon=1,
                                  min_epsilon=0.01,
                                  epsilon_decay=0.9999,
                                  eval_sleep=0.9,
                                  eval_frequency=1000,
                                  eval_episodes=100,
                                  train_log_frequency=100,
                                  eval_log_frequency=1,
                                  render=False,
                                  eval_render=False,
                                  video=False,
                                  video_fps=5,
                                  video_frequency=101,
                                  video_dir=default_output_dir() + "./results/videos/",
                                  gifs=False,
                                  gif_dir=default_output_dir() + "./results/gifs/",
                                  save_dir="./results/data/minimal_defense/",
                                  attacker=attacker,
                                  defender=not attacker,
                                  dqn_config=dqn_config,
                                  checkpoint_freq=300000)

    env_name = "idsgame-" + scenario + "-v3"
    env = gym.make(env_name, save_dir="./results/data/" + scenario + "/dqn/")

    agent = DQNAgent(env, q_agent_config, "")
    start = time.time()
    agent.train()
    print("*********Time to train*********: ", time.time() - start)

    train_result = agent.train_result
    eval_result = agent.eval_result
