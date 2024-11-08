'''
Functions for batching and unbatching actions/infos from swarms of agents
'''

import os
import torch
import numpy as np
from normalize import normalize_obs


def batchify_obs(obs, device, Agent):
    """Converts dict style observations to a multi-dimensional torch array."""
    obs = np.stack([obs[a]['obs'] for a in obs], axis=0)
    obs = normalize_obs(obs, Agent.get_observation_range())
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    return obs


def batchify(x, device):
    """Converts dict style returns and dones to batch of torch arrays."""
    x = np.stack([x[a] for a in x], axis=0) # convert to list of np arrays
    x = torch.tensor(x, dtype=torch.float32).to(device)
    return x


def unbatchify(x, env):
    """Converts torch array of returns and dones to dict style."""
    x = x.cpu().numpy()
    x = {a: np.expand_dims(x[i], 0) for i, a in enumerate(env.possible_agents)}
    return x


def clip_actions(actions):
    """Clips a set of actions to the range of [-1.0, 1.0]"""
    return torch.clip(actions, -1.0, 1.0)


def load_state_dicts(model_path, agents, device):
    """Loads the state dict of each agent with a model filepath or filepath to 
    a folder containing the models"""
    if (model_path):
        if (model_path.endswith('/')):
            print("Loading pre-existing models inside directory [" + model_path + "].")

            model_paths = [(model_path + file) for file in os.listdir(model_path) if file.endswith('.cleanrl_model')]
            assert (len(model_paths) == len(agents)), "Found " + str(len(model_paths)) + " agent models but require " + str(len(agents)) + " for this environment."

            for model_path, agent in zip(model_paths, agents):
                agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        else:
            print("Loading pre-existing model [" + model_path + "]")
            for agent in agents:
                agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
