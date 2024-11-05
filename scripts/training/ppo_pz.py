"""
Developed under the '2023 QDSA Collaborative Research Grant'.

Ownership is subject to the terms of the relevant QDSA Research Agreement.

The reproduction, distribution and utilization of this file as well as the 
communication of its contents to others without express authorization is 
prohibited. Offenders will be held liable for the payment of damages.
"""

'''
Trains a team of PPO agents in a pettingzoo style environment.

Utilises the 'shared network' approach where during a rollout a single policy is
utilised across all the agents and the trajectories are all combined to
train/update the policy weights.

Branched from CleanRL ppo_continuous_action.py at https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
'''

import os
import random
import traceback
import time
import timeit
import torch
import tyro
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Optional
from torch.utils.tensorboard import SummaryWriter

import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from batch_helpers import batchify_obs, batchify, unbatchify
from agents.jestel_agent import Agent
from godot import make_env


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    time_scale: float = 20.0
    """sets the simulator timescale"""
    model_path: Optional[str] = None
    """if a path is provided, will initialise the agent with the weights/biases of the model"""

    # Algorithm specific arguments
    file_path: Optional[str] = None
    """if a path is provided, will use the provided compiled executable"""
    no_graphics: bool = False
    """disables graphics for compiled environments"""
    total_timesteps: int = 2000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-5
    """the learning rate of the optimizer"""
    num_steps: int = 8192
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 64
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_updates: int = 0
    """the number of times the policy will update (computed in runtime)"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = make_env(args.time_scale, args.file_path, args.no_graphics, args.seed)
    action_space = env.action_space(env.possible_agents[0])[0]
    observation_space = env.observation_space(env.possible_agents[0])["obs"]

    agent = Agent(observation_space, action_space).to(device)
    if (args.model_path):
        print("Loading pre-existing model from [" + args.model_path + "]")
        agent.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    num_agents = len(env.possible_agents)
    obs = torch.zeros((args.num_steps, num_agents) + observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, num_agents) + action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, num_agents)).to(device)
    rewards = torch.zeros((args.num_steps, num_agents)).to(device)
    dones = torch.zeros((args.num_steps, num_agents)).to(device)
    values = torch.zeros((args.num_steps, num_agents)).to(device)

    # Derived arguments
    args.batch_size = int(num_agents * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.batch_size

    # Start the game
    global_step = 0
    episode_start_step = 0
    total_episodic_reward = 0
    start_time = time.time()
    next_obs = batchify_obs(env.reset()[0], device, Agent)
    next_done = torch.zeros(num_agents).to(device)

    try:
        for update in range(1, args.num_updates + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / args.num_updates
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += 1

                obs[step] = next_obs
                dones[step] = next_done

                # Get new action from the policy for this timestep
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                    actions[step] = action
                    logprobs[step] = logprob

                # Step the environment with the policy's chosen actions
                next_obs_unbatched, reward, next_done, _, infos = env.step(unbatchify(action, env))

                next_obs = batchify_obs(next_obs_unbatched, device, Agent)
                next_done = batchify(next_done, device).long()

                rewards[step] = batchify(reward, device).view(-1)
                total_episodic_reward += rewards[step]

                # Check for episode completion
                if (torch.prod(next_done) == 1):
                    episodic_msg = "global_step=" + str(global_step) + ", episodic_return=|"
                    for agent_index, agent_reward in enumerate(total_episodic_reward):
                        episodic_msg = episodic_msg + "{:.2f}".format(agent_reward.item()) + "|"
                        writer.add_scalar("charts/episodic_return(" + str(agent_index) + ")", agent_reward, global_step)
                    writer.add_scalar("charts/mean_episodic_return", torch.mean(total_episodic_reward), global_step)
                    writer.add_scalar("charts/episodic_length", (global_step - episode_start_step), global_step)
                    print(episodic_msg)

                    next_obs = batchify_obs(env.reset()[0], device, Agent)
                    total_episodic_reward = 0
                    episode_start_step = global_step

            # bootstrap value if not done
            optimization_start_time = timeit.default_timer()
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    # Calculate training metrics for debugging
                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)), ", Optimization: {:0.2e}s".format(timeit.default_timer() - optimization_start_time))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    except:
        print("Cancelling training run early due to exception:", traceback.print_exc(), "\n")
        pass

    model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
    torch.save(agent.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    env.close()
    writer.close()