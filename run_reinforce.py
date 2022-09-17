import os
import gym
import sys
import time

from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig
from gym_idsgame.agents.training_agents.policy_gradient.reinforce.reinforce import ReinforceAgent
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
    pg_agent_config = PolicyGradientAgentConfig(gamma=0.999,
                                                alpha_attacker=0.00001,
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
                                                num_episodes=200001,
                                                eval_render=False,
                                                gifs=False,
                                                gif_dir=default_output_dir() + "/results/gifs/",
                                                eval_frequency=10000,
                                                attacker=attacker,
                                                defender=not attacker,
                                                video_frequency=101,
                                                save_dir=default_output_dir() + "/results/data/" + scenario + '/',
                                                checkpoint_freq=5000,
                                                input_dim_attacker=88, #44
                                                output_dim_attacker=80, #40
                                                hidden_dim=64,
                                                num_hidden_layers=1, batch_size=32,
                                                gpu=False,
                                                tensorboard=False,
                                                tensorboard_dir=default_output_dir() + "/results/tensorboard/",
                                                optimizer="Adam",
                                                lr_exp_decay=False,
                                                lr_decay_rate=0.999)
    # Set up environment
    env_name = "idsgame-" + scenario + "-v3"
    env = gym.make(env_name, save_dir="./results/data/" + scenario + "/")

    # Set up agent
    attacker_agent = ReinforceAgent(env, pg_agent_config)
    start = time.time()
    attacker_agent.train()
    print("*********Time to train*********: ", time.time() - start)

    train_result = attacker_agent.train_result
    eval_result = attacker_agent.eval_result