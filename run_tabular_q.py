import os
import gym
import sys
import time
from gym_idsgame.agents.training_agents.q_learning.q_agent_config import QAgentConfig
from gym_idsgame.agents.training_agents.q_learning.tabular_q_learning.tabular_q_agent import TabularQAgent
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
    q_agent_config = QAgentConfig(gamma=0.999,
                                  alpha=0.0005,
                                  epsilon=1,
                                  render=False,
                                  eval_sleep=0.9,
                                  min_epsilon=0.01,
                                  eval_episodes=100,
                                  train_log_frequency=100,
                                  epsilon_decay=0.9999,
                                  video=False,
                                  eval_log_frequency=1,
                                  video_fps=5,
                                  video_dir=default_output_dir() + "/results/videos/",
                                  num_episodes=20001,
                                  eval_render=False,
                                  gifs=False,
                                  gif_dir=default_output_dir() + "/results/gifs/",
                                  eval_frequency=1000,
                                  attacker=attacker,
                                  defender=not attacker,
                                  video_frequency=101,
                                  save_dir=default_output_dir() + "/results/data/" + scenario + "/")

    env_name = "idsgame-" + scenario + "-v3"
    env = gym.make(env_name, save_dir=default_output_dir() + "/results/data/" + scenario + "/")

    attacker_agent = TabularQAgent(env, q_agent_config)
    start = time.time()
    attacker_agent.train()
    print("*********Time to train*********: ", time.time() - start)

    train_result = attacker_agent.train_result
    eval_result = attacker_agent.eval_result