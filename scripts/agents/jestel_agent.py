"""
Developed under the '2023 QDSA Collaborative Research Grant'.

Ownership is subject to the terms of the relevant QDSA Research Agreement.

The reproduction, distribution and utilization of this file as well as the 
communication of its contents to others without express authorization is 
prohibited. Offenders will be held liable for the payment of damages.
"""

"""
Implementation of the PPO agent network as described by "Obtaining Robust Control and Navigation Policies for 
Multi-robot Navigation via Deep Reinforcement Learning" (Jestel, et. al.)
"""

import numpy as np
import math
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class Agent(nn.Module):
    def __init__(self, envs):
        assert (envs.single_observation_space.shape[0] == Agent.observation_size()), ("The Jestel implementation requires an observation space of " + str(Agent.observation_size()) + " continuous values, received observation of shape: " + str(envs.single_observation_space.shape))
        assert (envs.single_action_space.shape[0] == Agent.action_size()), ("The Jestel implementation requires an action space of " + str(Agent.action_size()) + " continuous values, received action of shape: " + str(envs.single_action_space.shape))

        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    @staticmethod
    def observation_size():
        return 279 # See "Obtaining Robust Control and Navigation Policies for Multi-robot Navigation via Deep Reinforcement Learning", Jestel et. al

    @staticmethod
    def action_size():
        return 2 # See "Obtaining Robust Control and Navigation Policies for Multi-robot Navigation via Deep Reinforcement Learning", Jestel et. al

    @staticmethod
    def get_observation_range():
        ranges = np.array([
            [-6, 6],
            [0, 0.5],
            [-6, 6],
            [-6, 6],
            [0, 0.5],
            [-6, 6],
            [-math.pi, math.pi],
            [-2.1, 2.1],
            [-2.2, 2.2]
        ])

        # Add LiDAR scans
        num_scans = 270
        for i in range(num_scans):
            ranges = np.vstack([ranges, [0, 5]])

        print(ranges.shape)
        
        return ranges

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer