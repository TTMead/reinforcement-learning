"""
Developed under the '2023 QDSA Collaborative Research Grant'.

Ownership is subject to the terms of the relevant QDSA Research Agreement.

The reproduction, distribution and utilization of this file as well as the 
communication of its contents to others without express authorization is 
prohibited. Offenders will be held liable for the payment of damages.
"""

'''
Trains a team of PPO agents using a Unity3D environment passed through the
ML-Agents PettingZoo wrapper.

Utilises a 'federated learning' approach where during rollout each agent runs
its own training script that periodically shares the network weights to a
central server that aggregates and redistributes the combined network to each
agent.

Branched from CleanRL ppo_continuous_action.py at https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
'''

import random
import traceback
import time
from dataclasses import dataclass
from typing import Optional
import copy
import timeit

import supersuit as ss
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv

import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from agents.fire_agent import Agent

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
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder). Does nothing if env_id='unity'."""
    time_scale: float = 20.0
    """for Unity environments, sets the simulator timescale"""
    model_path: Optional[str] = None
    """if a path is provided, will initialise the agent with the weights/biases of the model"""

    # Algorithm specific arguments
    env_id: str = "unity"
    """the id of the gym environment, if set to 'unity' will attempt to connect to a Unity application over a default network port"""
    file_path: Optional[str] = None
    """if a path is provided, will use the provided compiled Unity executable"""
    no_graphics: bool = False
    """disables graphics from Unity3D environments"""
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

    share_batch_period: int = 3
    """the number of batches to run individually before sharing with the federated server"""
    share_batch_tao: float = 0.5
    """the sharing coefficient when merging network models. A value of 0.0 represents a hard averaging of all the network weights, while 1.0 represents no weight sharing between networks"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_updates: int = 0
    """the number of times the policy will update (computed in runtime)"""

def make_unity_env(editor_timescale, file_path, no_graphics, seed) -> UnityEnvironment:
    config_channel = EngineConfigurationChannel()
    if (not file_path):
        print("Waiting for Unity Editor on port " + str(UnityEnvironment.DEFAULT_EDITOR_PORT) + ". Press Play button now.")
    env = UnityEnvironment(seed=seed, file_name=file_path, no_graphics=no_graphics, side_channels=[config_channel])
    config_channel.set_configuration_parameters(time_scale=editor_timescale)
    env.reset()
    return env

def make_env(env_id, idx, capture_video, run_name, time_scale, gamma, file_path, no_graphics, seed):
    if (env_id == "unity"):
        unity_env = make_unity_env(time_scale, file_path, no_graphics, seed)
        env = UnityParallelEnv(unity_env)
    else:
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

    # For Unity PettingZoo wrapper, reset must be called to populate some env properties
    env.reset()
    env.metadata = {}
    assert (len(env._env.behavior_specs) == 1), ("Only single-team environments are currently supported, received " + str(len(env._env.behavior_specs)))

    # Add finite observation range to env from Agent (so normalize_obs_v0 will work)
    old_space = list(env._observation_spaces.values())[0]
    obs_range = np.tile(Agent.get_observation_range(), (Agent.stack_size(), 1))
    env._observation_spaces[list(env._observation_spaces.keys())[0]] = gym.spaces.Box(
                    low=obs_range[:,0],
                    high=obs_range[:,1],
                    shape=old_space.shape,
                    dtype=old_space.dtype)

    env = ss.flatten_v0(env)  # deal with dm_control's Dict observation space
    env = ss.clip_actions_v0(env)
    env = ss.normalize_obs_v0(env)
    # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env

def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    obs = np.stack([obs[a] for a in obs], axis=0) # convert to list of np arrays
    # obs = obs.transpose(0, -1, 1, 2) # transpose to be (batch, channel, height, width)
    obs = torch.tensor(obs).to(device) # convert to torch
    return obs

def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x

def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}

    return x

def average_models(models):
    avg_model = copy.deepcopy(models[0])
    
    # Get the state dictionaries of all models
    state_dicts = [model.state_dict() for model in models]
    
    # Calculate the average of each parameter
    avg_state_dict = {}
    for key in state_dicts[0].keys():
        avg_state_dict[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)
    
    # Load the averaged state dictionary into the new model
    avg_model.load_state_dict(avg_state_dict)
    
    return avg_model

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env = make_env(args.env_id, 0, args.capture_video, run_name, args.time_scale, args.gamma, args.file_path, args.no_graphics, args.seed)

    from gym.spaces.box import Box as legacy_box_type
    assert all(isinstance(env.action_space(agent), legacy_box_type) for agent in env.possible_agents), "only continuous action space is supported"

    # Convert all action spaces to float32 due to Unity ML-Agents bug [https://github.com/Unity-Technologies/ml-agents/issues/5976]
    for agent in env.possible_agents:
        env.action_space(agent).dtype = np.float32

    action_space = env.action_space(env.possible_agents[0])
    observation_space = env.observation_space(env.possible_agents[0])
    num_agents = len(env.possible_agents)

    agents = [Agent(observation_space, action_space).to(device) for i in range(num_agents)] 
    if (args.model_path):
        print("Loading pre-existing models inside directory [" + args.model_path + "].")

        model_paths = [(args.model_path + file) for file in os.listdir(args.model_path) if file.endswith('.cleanrl_model')]
        assert (len(model_paths) == len(agents)), "Found " + str(len(model_paths)) + " agent models but require " + str(len(agents)) + " for this environment."

        for model_path, agent in zip(model_paths, agents):
            agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))

    optimizers = [optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5) for agent in agents]

    # ALGO Logic: Storage setup
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

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    episode_start_step = 0
    total_episodic_reward = 0
    start_time = time.time()
    unity_error_count = 0
    steps_since_last_share = 0
    next_obs = batchify_obs(env.reset(), device)
    next_done = torch.zeros(num_agents).to(device)

    try:
        for update in range(1, args.num_updates + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / args.num_updates
                lrnow = frac * args.learning_rate
                for optimizer in optimizers: optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += 1

                obs[step] = next_obs
                dones[step] = next_done

                # Get the new action from each policy for this timestep
                new_values = torch.empty((0)).to(device)
                new_actions = torch.empty((0, 2)).to(device)
                new_logprobs = torch.empty((0)).to(device)
                with torch.no_grad():
                    for idx, agent in enumerate(agents):
                        action, logprob, _, value = agent.get_action_and_value(next_obs[idx].unsqueeze(0))
                        new_values = torch.cat((new_values, value.flatten()))
                        new_actions = torch.cat((new_actions, action))
                        new_logprobs = torch.cat((new_logprobs, logprob))
                values[step] = new_values
                actions[step] = new_actions
                logprobs[step] = new_logprobs

                # Step the environment with the batched policy actions
                next_obs_unbatched, reward, next_done, infos = env.step(unbatchify(new_actions, env))

                # A reward of -99.0 indicates a dead agent (workaround for Unity issues 
                # with indicating 'done' agents in parallel pettingzoo environments)
                dead_agent_reward = -99.0
                for agent_id, agent_reward in reward.items():
                    if agent_reward == dead_agent_reward:
                        # Assign any dead agents as 'done'
                        next_done[agent_id] = True

                        # GAE calculations assume a reward of 0 for dead agents
                        reward[agent_id] = 0

                next_obs = batchify_obs(next_obs_unbatched, device)
                next_done = batchify(next_done, device).long()

                # Update rewards
                rewards[step] = batchify(reward, device).view(-1)
                total_episodic_reward += rewards[step]

                # Unity will sometimes request both a decision step and termination step after stepping the environment.
                # This causes an error as PettingZoo==1.15.0 API assumes the same agent will not appear twice. If this
                # occurs, assume the level has completed.
                unity_error_occured = (len(env.aec_env.agents) > num_agents)
                if unity_error_occured:
                    print("Reseting early due to Unity error")
                    unity_error_count += 1

                # Check for episode completion
                if (torch.prod(next_done) == 1 or unity_error_occured):
                    episodic_msg = "global_step=" + str(global_step) + ", episodic_return=|"
                    for agent_index, agent_reward in enumerate(total_episodic_reward):
                        episodic_msg = episodic_msg + "{:.2f}".format(agent_reward.item()) + "|"
                        writer.add_scalar("charts/episodic_return(" + str(agent_index) + ")", agent_reward, global_step)
                    writer.add_scalar("charts/mean_episodic_return", torch.mean(total_episodic_reward), global_step)
                    writer.add_scalar("charts/episodic_length", (global_step - episode_start_step), global_step)
                    writer.add_scalar("charts/unity_error_count", unity_error_count, global_step)
                    print(episodic_msg)

                    next_obs = batchify_obs(env.reset(), device)
                    total_episodic_reward = 0
                    episode_start_step = global_step

            # bootstrap value if not done
            optimization_start_time = timeit.default_timer()
            with torch.no_grad():
                next_value = torch.cat([agent.get_value(next_obs[idx].unsqueeze(0)) for idx, agent in enumerate(agents)], dim=1)
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

            # Perform optimization for each agent on the collected data
            for idx, agent in enumerate(agents):
                b_obs = obs[:,idx]
                b_logprobs = logprobs[:,idx]
                b_actions = actions[:,idx]
                b_advantages = advantages[:,idx]
                b_returns = returns[:,idx]
                b_values = values[:,idx]

                # Optimizing the policy and value network
                b_inds = np.arange(args.num_steps)
                clipfracs = []
                for epoch in range(args.update_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, args.num_steps, args.minibatch_size):
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

                        optimizers[idx].zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        optimizers[idx].step()

                    if args.target_kl is not None and approx_kl > args.target_kl:
                        break

                y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                writer.add_scalar("charts/learning_rate(" + str(idx) + ")", optimizers[idx].param_groups[0]["lr"], global_step)
                writer.add_scalar("losses/value_loss(" + str(idx) + ")", v_loss.item(), global_step)
                writer.add_scalar("losses/policy_loss(" + str(idx) + ")", pg_loss.item(), global_step)
                writer.add_scalar("losses/entropy(" + str(idx) + ")", entropy_loss.item(), global_step)
                writer.add_scalar("losses/old_approx_kl(" + str(idx) + ")", old_approx_kl.item(), global_step)
                writer.add_scalar("losses/approx_kl(" + str(idx) + ")", approx_kl.item(), global_step)
                writer.add_scalar("losses/clipfrac(" + str(idx) + ")", np.mean(clipfracs), global_step)
                writer.add_scalar("losses/explained_variance(" + str(idx) + ")", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)), ", Optimization: {:0.2e}s".format(timeit.default_timer() - optimization_start_time))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # Combine networks at the central server
            steps_since_last_share += 1
            if (steps_since_last_share >= args.share_batch_period):
                averaging_start_time = timeit.default_timer()
                steps_since_last_share = 0

                hard_avg_model = average_models(agents)
                hard_avg_dict = hard_avg_model.state_dict()

                # Scale each agent towards the hard average
                for agent in agents:
                    state_dict = agent.state_dict()
                    new_state_dict = {}
                    for key in state_dict.keys():
                        new_state_dict[key] = args.share_batch_tao * state_dict[key] + (1 - args.share_batch_tao) * hard_avg_dict[key]
                    
                    agent.load_state_dict(new_state_dict)
                
                print("Averaging network weights: {:0.2e}s".format(timeit.default_timer() - averaging_start_time))
    except:
        print("Cancelling training run early due to exception:", traceback.print_exc(), "\n")
        pass

    env.close()

    for idx, agent in enumerate(agents):
        model_path = f"runs/{run_name}/{args.exp_name}{idx}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
    print(f"\nModels saved to {model_path}")

    import pickle 
    metadata_path = f"runs/{run_name}/env_metadata.pkl"
    with open(metadata_path, 'wb') as metadata_file:
        pickle.dump(env.metadata, metadata_file)

    import json, dataclasses
    metadata_path = f"runs/{run_name}/env_metadata.json"
    json_path = f"runs/{run_name}/args.json"
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(dataclasses.asdict(args), file, ensure_ascii=False, indent=4)

    with open(metadata_path, 'w', encoding='utf-8') as file:
        json.dump(env.metadata, file, ensure_ascii=False, indent=4)

    writer.close()