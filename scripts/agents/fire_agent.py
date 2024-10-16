"""
Developed under the '2023 QDSA Collaborative Research Grant'.

Ownership is subject to the terms of the relevant QDSA Research Agreement.

The reproduction, distribution and utilization of this file as well as the 
communication of its contents to others without express authorization is 
prohibited. Offenders will be held liable for the payment of damages.
"""

"""
Implementation of the PPO agent network for the QDSA Fire Fighting scenario.
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

        self.velocity_stream = nn.Sequential(
            nn.Linear(Agent.stack_size() * 2, 32),
            nn.ReLU()
        )

        self.fire_stream = nn.Sequential(
            nn.Linear(Agent.stack_size() * 12, 48 * 4),
            nn.ReLU()
        )

        self.agent_stream = nn.Sequential(
            nn.Linear(Agent.stack_size() * 9, 48 * 3),
            nn.ReLU()
        )

        output_layer_size = 624
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
        o_v = observation[:,:,270:272].reshape(batch_size, -1)
        o_f = observation[:,:,272:284].reshape(batch_size, -1)
        o_a = observation[:,:,284:293].reshape(batch_size, -1)

        assert (o_l.shape == (batch_size, Agent.stack_size(), 270)), "Lidar observation should have a shape of (batch_size, " + str(Agent.stack_size()) + ", 270), received " + str(o_l.shape)
        assert (o_v.shape == (batch_size, Agent.stack_size() * 2)), "Velocity observation should have a shape of (batch_size, " + str(Agent.stack_size() * 2) + "), received " + str(o_v.shape)
        assert (o_f.shape == (batch_size, Agent.stack_size() * 12)), "Fire observation should have a shape of (batch_size, " + str(Agent.stack_size() * 12) + "), received " + str(o_f.shape)
        assert (o_a.shape == (batch_size, Agent.stack_size() * 9)), "Agent observation should have a shape of (batch_size, " + str(Agent.stack_size() * 9) + "), received " + str(o_a.shape)

        # Create network branches
        out1 = self.lidar_stream(o_l)
        out2 = self.velocity_stream(o_v)
        out3 = self.fire_stream(o_f)
        out4 = self.agent_stream(o_a)

        # Concatenate into output hidden layer
        combined = torch.cat((out1, out2, out3, out4), dim=1)
        assert (combined.shape == (batch_size, 624)), "Concatenated hidden layer should have a shape of (batch_size, 624), received " + str(combined.shape)

        output = self.output_stream(combined)
        return output

class Agent(nn.Module):
    def __init__(self, observation_space, action_space):
        assert (observation_space.shape[0] == Agent.stacked_observation_size()), ("The Jestel implementation requires an observation space of " + str(Agent.stacked_observation_size()) + " continuous values, received observation of shape: " + str(observation_space.shape[0]))
        assert (action_space.shape[0] == Agent.action_size()), ("The Jestel implementation requires an action space of " + str(Agent.action_size()) + " continuous values, received action of shape: " + str(action_space.shape[0]))

        super().__init__()

        self.critic = JestelNetwork(output_size=1)
        self.actor_mean = JestelNetwork(output_size=Agent.action_size())
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_space.shape)))

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
        return 4

    @staticmethod
    def observation_size():
        """Number of float values generated by the agent per timestep."""
        return 293

    @staticmethod
    def stacked_observation_size():
        return Agent.stack_size() * Agent.observation_size()

    @staticmethod
    def action_size():
        """Number of continuous actions the agent must provide a value for each timestep."""
        return 2

    @staticmethod
    def get_observation_range():
        ranges = np.empty((0,2))

        # O_l (LiDAR) scan Ranges
        num_scans = 270
        for i in range(num_scans):
            ranges = np.vstack([ranges, [0, 5]])
        
        # O_v (robot velocities) scan ranges
        ranges = np.vstack([ranges, [-1.1, 2.1]])
        ranges = np.vstack([ranges, [-2.1, 2.1]])

        # O_f (fire state) (rel_x, rel_y, dist) ranges
        num_fires = 4
        for i in range(num_fires):
            ranges = np.vstack([ranges, [-20, 20]])
            ranges = np.vstack([ranges, [-20, 20]])
            ranges = np.vstack([ranges, [0, 40]])

        # O_a (agent state) (rel_x, rel_y, dist) ranges
        num_agents = 4
        for i in range(num_agents - 1):
            ranges = np.vstack([ranges, [-20, 20]])
            ranges = np.vstack([ranges, [-20, 20]])
            ranges = np.vstack([ranges, [0, 40]])

        return ranges

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer