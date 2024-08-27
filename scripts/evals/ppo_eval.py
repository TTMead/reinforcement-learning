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
from ppo import Agent, make_unity_env
from normalize import StaticNormalizeObservation
import numpy as np

def make_env(env_id, idx, capture_video, run_name, time_scale, gamma, observation_mean, observation_var):
    def thunk():
        if (env_id == "unity"):
            unity_env = make_unity_env(time_scale)
            env = UnityToGymWrapper(unity_env, uint8_visual=False, flatten_branched=False, allow_multiple_obs=False)
        else:
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = StaticNormalizeObservation(env, observation_mean, observation_var)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk

def evaluate(
    model_path: str,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    gamma: float = 0.99,
    metadata_path: str = None
):
    # Create gym environment
    time_scale = 2.0
    if (metadata_path):
        # If metadata is provided, load in the saved mean/std
        import pickle
        with open(metadata_path, 'rb') as metadata_file:
            env_metadata = pickle.load(metadata_file)

            print(env_metadata["observation_mean"])
            print(env_metadata["observation_var"])

            assert (env_metadata["observation_mean"] is not None), "Metadata does not contain observation mean"
            assert (env_metadata["observation_var"] is not None), "Metadata does not contain observation variance"

            mean = env_metadata["observation_mean"]
            var = env_metadata["observation_var"]
    else:
        mean = [-0.28141704,0.25,1.02541261,0.05212049,0.26647945,-0.17269537,-0.1262586,-0.4684438,0.02271854]
        var = [6.51560695e+00,1.04929981e-09,5.42513869e+00,1.62295415e+00,2.75232264e-04,1.70809007e+00,6.15432472e+00,9.77105898e-01,8.06864739e-01]

    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, time_scale, gamma, mean, var)])

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
    metadata_path: Optional[str] = None
    env_id: str = "unity"


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
        eval_episodes=50,
        run_name=f"eval",
        Model=Agent,
        device="cpu",
        capture_video=False,
        metadata_path=args.metadata_path
    )