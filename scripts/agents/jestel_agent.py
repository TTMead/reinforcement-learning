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
            nn.Conv1d(in_channels=Agent.stack_size(), out_channels=16, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * ((((lidar_count - 7) // 3 + 1) - 5) // 2 + 1), 256),
            nn.ReLU()
        )

        self.direction_stream = nn.Sequential(
            nn.Linear(Agent.stack_size() * 2, 32),
            nn.ReLU()
        )

        self.distance_stream = nn.Sequential(
            nn.Linear(Agent.stack_size() * 1, 16),
            nn.ReLU()
        )

        self.velocity_stream = nn.Sequential(
            nn.Linear(Agent.stack_size() * 2, 32),
            nn.ReLU()
        )

        output_layer_size = 256+32+16+32
        self.output_stream = nn.Sequential(
            nn.Linear(output_layer_size, 384),
            nn.ReLU(),
            nn.Linear(384, output_size)
        )
    
    def forward(self, raw_observation):
        batch_size = raw_observation.shape[0]
        assert (raw_observation.shape[1] == Agent.stacked_observation_size()), "Jestel agent should receive a raw observation of shape (batch_size, " + str(Agent.stacked_observation_size()) + "), received " + str(raw_observation.shape)
        observation = raw_observation.reshape(batch_size, Agent.stack_size(), Agent.observation_size())

        # Split observation into components
        o_l = observation[:,:,0:270]
        o_g = observation[:,:,270:272].reshape(batch_size, -1)
        o_d = observation[:,:,272].unsqueeze(2).reshape(batch_size, -1)
        o_v = observation[:,:,273:275].reshape(batch_size, -1)

        assert (o_l.shape == (batch_size, Agent.stack_size(), 270)), "Lidar observation should have a shape of (batch_size, " + str(Agent.stack_size()) + ", 270), received " + str(o_l.shape)
        assert (o_g.shape == (batch_size, Agent.stack_size() * 2)), "Goal direction observation should have a shape of (batch_size, " + str(Agent.stack_size() * 2) + "), received " + str(o_g.shape)
        assert (o_d.shape == (batch_size, Agent.stack_size() * 1)), "Goal distance observation should have a shape of (batch_size, " + str(Agent.stack_size() * 1) + "), received " + str(o_d.shape)
        assert (o_v.shape == (batch_size, Agent.stack_size() * 2)), "Velocity observation should have a shape of (batch_size, " + str(Agent.stack_size() * 2) + "), received " + str(o_v.shape)

        # Create network branches
        out1 = self.lidar_stream(o_l)
        out2 = self.direction_stream(o_g)
        out3 = self.distance_stream(o_d)
        out4 = self.velocity_stream(o_v)

        # Concatenate into output hidden layer
        combined = torch.cat((out1, out2, out3, out4), dim=1)
        assert (combined.shape == (batch_size, 336)), "Concatenated hidden layer should have a shape of (batch_size, 336), received " + str(combined.shape)

        output = self.output_stream(combined)
        return output

class Agent(nn.Module):
    def __init__(self, envs):
        assert (envs.single_observation_space.shape[0] == Agent.stacked_observation_size()), ("The Jestel implementation requires an observation space of " + str(Agent.stacked_observation_size()) + " continuous values, received observation of shape: " + str(envs.single_observation_space.shape))
        assert (envs.single_action_space.shape[0] == Agent.action_size()), ("The Jestel implementation requires an action space of " + str(Agent.action_size()) + " continuous values, received action of shape: " + str(envs.single_action_space.shape))

        super().__init__()

        self.critic = JestelNetwork(output_size=1)
        self.actor_mean = JestelNetwork(output_size=Agent.action_size())
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
    def stack_size():
        """The number of sequential observation vectors for the agent to receive."""
        return 4 # See "Obtaining Robust Control and Navigation Policies for Multi-robot Navigation via Deep Reinforcement Learning", Jestel et. al

    @staticmethod
    def observation_size():
        """Number of float values generated by the agent per timestep."""
        return 275 # See "Obtaining Robust Control and Navigation Policies for Multi-robot Navigation via Deep Reinforcement Learning", Jestel et. al

    @staticmethod
    def stacked_observation_size():
        return Agent.stack_size() * Agent.observation_size()

    @staticmethod
    def action_size():
        """Number of continuous actions the agent must provide a value for each timestep."""
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
        ranges = np.vstack([ranges, [0, 30]])

        # O_v (robot velocities) scan ranges
        ranges = np.vstack([ranges, [-1.1, 2.1]])
        ranges = np.vstack([ranges, [-2.1, 2.1]])

        return ranges

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer