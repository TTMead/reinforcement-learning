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

from typing import Callable, Optional
from dataclasses import dataclass

import gymnasium as gym
import torch

import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from unity_gymnasium_env import UnityToGymWrapper
from training.ppo import Agent, make_env
from normalize import MinMaxNormalizeObservation
import numpy as np

def evaluate(
    model_path: str,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    gamma: float = 0.99,
    timescale: float = 1.0
):
    # Create gym environment
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, timescale, gamma)])

    # Load model weights from disk
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    # Run evaluation
    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns

@dataclass
class Args:
    model_path: Optional[str]
    """the path of the model weights to load in for evaluation. Will download a model from huggingface if no path is provided"""
    env_id: str = "unity"
    """the id of the gym environment, if set to 'unity' will attempt to connect to a Unity application over a default network port"""
    time_scale: float = 20.0
    """for Unity environments, sets the simulator timescale"""
    eval_episodes: int = 100
    """the number of episodes to run for evaluation"""


if __name__ == "__main__":
    import tyro
    args = tyro.cli(Args)

    if (args.model_path):
        model_path = args.model_path
    else:
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id="sdpkjc/Hopper-v4-ppo_continuous_action-seed1", filename="ppo_continuous_action.cleanrl_model"
        )
    
    evaluate(
        model_path,
        args.env_id,
        eval_episodes=args.eval_episodes,
        run_name=f"eval",
        Model=Agent,
        device="cpu",
        capture_video=False,
        gamma=0.99,
        timescale=args.time_scale
    )