"""
Developed under the '2023 QDSA Collaborative Research Grant'.

Ownership is subject to the terms of the relevant QDSA Research Agreement.

The reproduction, distribution and utilization of this file as well as the 
communication of its contents to others without express authorization is 
prohibited. Offenders will be held liable for the payment of damages.
"""

'''
Branched from CleanRL ppo_eval.py at https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl_utils/evals/ppo_eval.py
'''

import tyro
import torch
import numpy as np
from typing import Optional
from dataclasses import dataclass

import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from batch_helpers import batchify_obs, batchify, unbatchify, load_state_dicts
from agents.jestel_agent import Agent
from godot import make_env


@dataclass
class Args:
    model_path: str
    """the path of the model weights to load in for evaluation. Will download a model from huggingface if no path is provided"""
    time_scale: float = 20.0
    """for Unity environments, sets the simulator timescale"""
    eval_episodes: int = 100
    """the number of episodes to run for evaluation"""
    no_graphics: bool = False
    """disables graphics for compiled environments"""
    file_path: Optional[str] = None
    """if a path is provided, will use the provided compiled executable"""
    seed: int = 1
    """seed of the experiment"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(args.time_scale, args.file_path, args.no_graphics, args.seed)
    action_space = env.action_space(env.possible_agents[0])[0]
    observation_space = env.observation_space(env.possible_agents[0])["obs"]

    num_agents = len(env.possible_agents)
    agents = [Agent(observation_space, action_space).to(device) for i in range(num_agents)] 
    print("Loading pre-existing models inside directory [" + args.model_path + "].")

    model_paths = [(args.model_path + file) for file in os.listdir(args.model_path) if file.endswith('.cleanrl_model')]
    assert (len(model_paths) == len(agents)), "Found " + str(len(model_paths)) + " agent models but require " + str(len(agents)) + " for this environment."

    for model_path, agent in zip(model_paths, agents):
        agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))

    total_episodic_reward = 0
    global_step = 0
    next_obs = batchify_obs(env.reset()[0], device)
    episode_count = 0
    episodic_rewards = []
    while episode_count < args.eval_episodes:
        global_step += 1

        actions = torch.empty((0, 2)).to(device)
        with torch.no_grad():
            for idx, agent in enumerate(agents):
                action, _, _, _ = agent.get_action_and_value(next_obs[idx].unsqueeze(0))
                actions = torch.cat((actions, action))

        next_obs_unbatched, reward, next_done, infos = env.step(unbatchify(actions, env))

        next_obs = batchify_obs(next_obs_unbatched, device)
        next_done = batchify(next_done, device).long()

        total_episodic_reward += batchify(reward, device).view(-1)

        # Check for episode completion
        if (torch.prod(next_done) == 1):
            episodic_msg = "global_step=" + str(global_step) + ", episodic_return=|"
            for agent_index, agent_reward in enumerate(total_episodic_reward):
                episodic_msg = episodic_msg + "{:.2f}".format(agent_reward.item()) + "|"
                episodic_rewards.append(agent_reward)
            print(episodic_msg)

            next_obs = batchify_obs(env.reset()[0], device)
            total_episodic_reward = 0
            episode_count += 1
    
    # Calculate episodic reward stats.
    pct_positive = sum(reward > 0 for reward in episodic_rewards) / len(episodic_rewards)
    print("Percentage of positive reward episodes: " + str(pct_positive))

    env.close()