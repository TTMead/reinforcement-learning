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

from typing import Optional
from dataclasses import dataclass

import torch

import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from training.ppo_federated import Agent, make_env, batchify_obs, unbatchify, batchify
import numpy as np

@dataclass
class Args:
    model_path: str
    """the path of the model weights to load in for evaluation. Will download a model from huggingface if no path is provided"""
    env_id: str = "unity"
    """the id of the gym environment, if set to 'unity' will attempt to connect to a Unity application over a default network port"""
    time_scale: float = 20.0
    """for Unity environments, sets the simulator timescale"""
    eval_episodes: int = 100
    """the number of episodes to run for evaluation"""
    no_graphics: bool = False
    """disables graphics from Unity3D environments"""
    file_path: Optional[str] = None
    """if a path is provided, will use the provided compiled Unity executable"""
    seed: int = 1
    """seed of the experiment"""


if __name__ == "__main__":
    import tyro
    args = tyro.cli(Args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"eval"
    capture_video = False
    gamma = 0.99
    env = make_env(args.env_id, 0, capture_video, run_name, args.time_scale, gamma, args.file_path, args.no_graphics, args.seed)

    # Convert all action spaces to float32 due to Unity ML-Agents bug [https://github.com/Unity-Technologies/ml-agents/issues/5976]
    for agent in env.possible_agents:
        env.action_space(agent).dtype = np.float32

    action_space = env.action_space(env.possible_agents[0])
    observation_space = env.observation_space(env.possible_agents[0])
    num_agents = len(env.possible_agents)

    # Load model weights from disk
    agents = [Agent(observation_space, action_space).to(device) for i in range(num_agents)] 
    print("Loading pre-existing models inside directory [" + args.model_path + "].")

    model_paths = [(args.model_path + file) for file in os.listdir(args.model_path) if file.endswith('.cleanrl_model')]
    assert (len(model_paths) == len(agents)), "Found " + str(len(model_paths)) + " agent models but require " + str(len(agents)) + " for this environment."

    for model_path, agent in zip(model_paths, agents):
        agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))


    # Run evaluation
    total_episodic_reward = 0
    unity_error_count = 0
    global_step = 0
    next_obs = batchify_obs(env.reset(), device)
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

        # A reward of -99.0 indicates a dead agent (workaround for Unity issues 
        # with indicating 'done' agents in parallel pettingzoo environments)
        dead_agent_reward = -99.0
        for agent_id, agent_reward in reward.items():
            if agent_reward == dead_agent_reward:
                reward[agent_id] = 0
        

        next_obs = batchify_obs(next_obs_unbatched, device)
        next_done = batchify(next_done, device).long()

        # Update rewards
        total_episodic_reward += batchify(reward, device).view(-1)

        # Unity will sometimes request both a decision step and termination step after stepping the environment.
        # This causes an error as PettingZoo==1.15.0 API assumes the same agent will not appear twice. If this
        # occurs, assume the level has completed.
        agents_have_reset = (len(env.aec_env.agents) > num_agents)
        if agents_have_reset:
            print("Early agent reset")
            unity_error_count += 1

        # Check for episode completion
        if (torch.prod(next_done) == 1 or agents_have_reset):
            episodic_msg = "global_step=" + str(global_step) + ", episodic_return=|"
            for agent_index, agent_reward in enumerate(total_episodic_reward):
                episodic_msg = episodic_msg + "{:.2f}".format(agent_reward.item()) + "|"
                episodic_rewards.append(agent_reward)
            print(episodic_msg)

            next_obs = batchify_obs(env.reset(), device)
            total_episodic_reward = 0
            episode_count += 1
    
    # Calculate episodic reward stats.
    pct_positive = sum(reward > 0 for reward in episodic_rewards) / len(episodic_rewards)
    print("Percentage of positive reward episodes: " + str(pct_positive))

    env.close()