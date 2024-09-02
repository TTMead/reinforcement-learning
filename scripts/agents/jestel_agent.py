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
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class JestelNetwork(nn.Module):
    def __init__(self, output_size):
        super(JestelNetwork, self).__init__()
        
        lidar_count = 270
        self.lidar_stream = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * ((((lidar_count - 7) // 3 + 1) - 5) // 2 + 1), 256),
            nn.ReLU(),
            nn.Flatten(start_dim=0),
        )

        self.direction_stream = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )

        self.distance_stream = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )

        self.velocity_stream = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )

        output_layer_size = 256+32+16+32
        self.output_stream = nn.Sequential(
            nn.Linear(output_layer_size, 384),
            nn.ReLU(),
            nn.Linear(384, output_size)
        )
    
    def forward(self, o):
        # Split observation into components
        o_l = o[0,0:270]
        o_g = o[0,270:272]
        o_d = o[0,272]
        o_v = o[0,273:275]

        # Create network branches
        out1 = self.lidar_stream(o_l.unsqueeze(0).unsqueeze(0))
        out2 = self.direction_stream(o_g)
        out3 = self.distance_stream(o_d.unsqueeze(0))
        out4 = self.velocity_stream(o_v)

        # Concatenate into output hidden layer
        combined = torch.cat((out1, out2, out3, out4), dim=0)
        output = self.output_stream(combined)
        
        return output.unsqueeze(0)

class Agent(nn.Module):
    def __init__(self, envs):
        assert (envs.single_observation_space.shape[0] == Agent.observation_size()), ("The Jestel implementation requires an observation space of " + str(Agent.observation_size()) + " continuous values, received observation of shape: " + str(envs.single_observation_space.shape))
        assert (envs.single_action_space.shape[0] == Agent.action_size()), ("The Jestel implementation requires an action space of " + str(Agent.action_size()) + " continuous values, received action of shape: " + str(envs.single_action_space.shape))

        super().__init__()

        self.critic = JestelNetwork(output_size=1)
        self.actor_mean = JestelNetwork(output_size=2)
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
        observation_size = 275
        stack_size = 4
        return (observation_size * stack_size) # See "Obtaining Robust Control and Navigation Policies for Multi-robot Navigation via Deep Reinforcement Learning", Jestel et. al

    @staticmethod
    def action_size():
        return 2 # See "Obtaining Robust Control and Navigation Policies for Multi-robot Navigation via Deep Reinforcement Learning", Jestel et. al

    @staticmethod
    def get_observation_range():
        ranges = np.empty((0,2))

        # O_l (LiDAR) scan Ranges
        num_scans = 270
        for i in range(num_scans):
            ranges = np.vstack([ranges, [0, 5]])

        # O_g (direction to goal) scan ranges
        ranges = np.vstack([ranges, [-1, 1]])
        ranges = np.vstack([ranges, [-1, 1]])

        # O_d (distance to goal) scan range
        ranges = np.vstack([ranges, [-20, 20]])

        # O_v (robot velocities) scan ranges
        ranges = np.vstack([ranges, [-2.1, 2.1]])
        ranges = np.vstack([ranges, [-2.1, 2.1]])

        return ranges

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer