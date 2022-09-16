import os
import gym
import time
import sys
from gym_idsgame.agents.training_agents.q_learning.abstract_qhd_agent_config import AbstractQHDAgentConfig
from gym_idsgame.agents.training_agents.q_learning.qhd.qhd import QHDAgent
from gym_idsgame.agents.training_agents.q_learning.qhd.qhd_config import QHDConfig
from experiments.util import util

# def get_script_path():
#     """
#     :return: the script path
#     """
#     return os.path.dirname(os.path.realpath(sys.argv[0]))


# def default_output_dir() -> str:
#     """
#     :return: the default output dir
#     """
#     script_dir = get_script_path()
#     return script_dir

# Program entrypoint
if __name__ == '__main__':
    random_seed = 0
    util.create_artefact_dirs('./', random_seed)

    for lr in [0.00001]: #, 0.0001, 0.001, 0.01]:
        qhd_config = QHDConfig(input_dim=88,
                               defender_output_dim=88,  # attacker would need 80: 10 attacks (+1 for defender), 8 nodes
                               attacker_output_dim=80,  # attacker would need 80: 10 attacks (+1 for defender), 8 nodes
                               replay_memory_size=10000,
                               batch_size=32,
                               target_network_update_freq=250,  # TODO: Hyperparameter for fine-tuning
                               gpu=False,
                               tensorboard=True,
                               tensorboard_dir="./results/tensorboard/",
                               lr_exp_decay=False,
                               lr_decay_rate=0.9999)

        qhd_agent_config = AbstractQHDAgentConfig(gamma=0.999,
                                      lr=lr,  # i.e. learning rate
                                      num_episodes=20001,
                                      epsilon=1,
                                      min_epsilon=0.01,
                                      epsilon_decay=0.95,  # TODO: Hyperparameter for fine-tuning
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
                                      video_dir="./results/videos/",
                                      gifs=False,
                                      gif_dir="./results/gifs/",
                                      save_dir="./results/data/random_defense/",
                                      attacker=True,
                                      defender=False,
                                      qhd_config=qhd_config,
                                      checkpoint_freq=20000)

        # Set up environment
        env_name = "idsgame-random_defense-v3"
        env = gym.make(env_name, save_dir="./results/data/random_defense/")

        # Set up agent
        agent = QHDAgent(env, qhd_agent_config, "FINAL_")

        start = time.time()
        agent.train()
        print("*********Time to train*********: ", time.time() - start)

        # TODO: I need to implement these functions
        train_result = agent.train_result
        eval_result = agent.eval_result