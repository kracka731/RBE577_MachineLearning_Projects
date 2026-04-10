import os
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import math

from helpers.metrics import MetricsTracker
from helpers.utils import (get_network_input_shape, get_screen, 
                           make_env, setup_camera, get_kuka_action_dim)
from lib.a3c.model import ActorCritic
from lib.a3c.objectives import (
    compute_actor_loss,
    compute_advantage,
    compute_bootstrapped_returns,
    compute_critic_loss,
)


def emit_log(message, log_path=None):
    """Print a training message and optionally append it to the run log."""
    print(message, flush=True)

    if log_path is None:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"{timestamp} - INFO - {message}\n")




def worker_process(
    worker_id,
    global_net,
    optimizer,
    global_ep,
    max_episodes,
    lock,
    config,
    device,
    shared_stats,
    log_path=None,
):
    """Run one A3C worker."""
    env = make_env(config, worker_id)

    # TODO: Create the worker's local actor-critic network
    state_dim = get_network_input_shape(config)
    action_dim = get_kuka_action_dim(config)
    net = config['network']

    local_net = ActorCritic(state_dim, action_dim, net["shared_layers"], 
                            net["critic_hidden_layers"], net["actor_hidden_layers"], 
                            init_type=net["init_type"])
    local_net.to(device)

    # TODO: Synchronize the local worker network with the shared global network
    local_net.load_state_dict(global_net.state_dict())  # Replace with your implementation

    metrics = MetricsTracker()

    gamma = config["hyperparameters"]["gamma"]
    t_max = config["hyperparameters"]["t_max"]
    entropy_coef = config["hyperparameters"]["entropy_coef"]
    value_loss_coef = config["hyperparameters"]["value_loss_coef"]
    grad_clip = config["hyperparameters"]["grad_clip"]
    log_interval = config["logging"]["log_interval"]
    save_interval = config["logging"]["save_interval"]

    env.reset()
    setup_camera(env, config)
    state = get_screen(env, device, config)
    episode_reward = 0.0
    episode_steps = 0

    while True:
        with global_ep.get_lock():
            if global_ep.value >= max_episodes:
                break

        # TODO: Refresh local parameters and clear stale gradients
        # Hint: Each rollout should start from the newest shared weights, and the
        # local worker model should not carry old gradients into the next update.
        optimizer.zero_grad()
        env.reset()  # Replace with your implementation
        setup_camera(env, config)
        state = get_screen(env, device, config)
        #FIXME: probably missing things here

        # log_probs = torch.empty((1, action_dim)).to(device)
        values = []
        rewards = []
        log_prob_list = []
        entropy_list = []

        # entropies = torch.empty((1, action_dim)).to(device)
        done = False

        for _ in range(t_max):
            # TODO: Run the local network to get the current policy output and value estimate
            # Hint: The model returns the actor output and critic value; then use the
            # model helper to turn the actor output into a distribution.
            action_loc, value = local_net(state)  # Replace with your implementation
            dist, entropy = local_net.get_action_distribution(action_loc)  # Replace with your implementation
            
            # debug - prevent exploding vals
            if torch.isnan(action_loc).any():
                print("NaN detected in action_loc")
                break

            # TODO: Sample an action and compute the policy terms needed later
            # action = dist.rsample()  
            # entropy = compute_entropy(dist)
            action = dist.sample()
            action = torch.clamp(action, -1 + 1e-6, 1 - 1e-6).flatten()
            log_prob = dist.log_prob(action)  
            # print(f"log_prob: {log_prob.detach()}")
            # entropy = log_prob
            # print(f"entropy: {entropy}")
            action_np = None  # Replace with your implementation

            # TODO: Step the environment and preprocess the next observation
            observation, reward, done, debug = env.step(action)
            next_state = get_screen(env, device, config)  # Replace with your implementation

            # TODO: Save the rollout information needed for the loss computation
            # Hint: Store the policy terms, value estimates, and rewards one step at a time.
            # log_probs = torch.vstack([log_probs, log_prob])
            log_prob_list.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropy_list.append(entropy)
            # entropies = torch.vstack([entropies, entropy])

            episode_reward += reward
            episode_steps += 1
            state = next_state

            if done:
                break

        with torch.no_grad():
            # TODO: Compute the bootstrap value at the rollout boundary
            # Hint: If the episode ended, the bootstrap target should be zero.
            # Otherwise, use the local critic to estimate the unfinished tail.
            if not done:
                bootstrap_value = 0
            else:
                bootstrap_value = float(reward + gamma * value * (1-done))
                # FIXME additional computations
            values.append(value + bootstrap_value)

        # TODO: Convert the rollout into batched tensors and objectives
        return_batch = compute_bootstrapped_returns(rewards, gamma, bootstrap_value) 
        log_prob_batch = torch.stack(log_prob_list)
        value_batch = torch.tensor(values).squeeze(-1)
        entropy_batch = torch.stack(entropy_list)
        
        advantage_batch = compute_advantage(return_batch, value_batch.detach()).to(device) 
        actor_loss = compute_actor_loss(log_prob_batch, advantage_batch.detach(), entropy_batch, entropy_coef)
        critic_loss = compute_critic_loss(return_batch.detach(), value_batch)
        total_loss = actor_loss + value_loss_coef*critic_loss  

        # FIXME: i didnt write these next 4 lines but what are they???
        actor_loss_value = actor_loss.item()
        critic_loss_value = critic_loss.item()
        total_loss_value = total_loss.item()
        policy_std_value = float(F.softplus(local_net.sigma).mean().item())
        # print(f"Losses. Actor {actor_loss_value} | Critic {critic_loss_value} | Tot {total_loss_value} | STD {policy_std_value}")

        # TODO: Backpropagate through the local worker model and clip gradients
        # Hint: Gradients are still computed on the worker's local network first.
        # FIXME ???
        optimizer.zero_grad()
        total_loss.backward()
        for local_param, global_param in zip(local_net.parameters(), global_net.parameters()):
            global_param._grad = local_param.grad
        torch.nn.utils.clip_grad_norm_(local_net.parameters(), grad_clip)
        # torch.nn.utils.clip_grad_norm_(global_net.parameters(), grad_clip)
        optimizer.step()

        # TODO: Apply the shared update inside a synchronized section
        # FIXME ???
        local_net.load_state_dict(global_net.state_dict())

        current_ep = global_ep.value
        if done:
            metrics.add_episode_reward(episode_reward)
            metrics.add_loss(total_loss_value)
            metrics.add_episode_length(episode_steps)

            # TODO: Update the shared episode counter and logging stats
            with global_ep.get_lock(): # FIXME with lock vs with global_ep.get_lock()???
                global_ep.value = global_ep.value + 1
                if global_ep.value % 25 == 0:
                    print(f"GLOBAL EP: {global_ep.value}")
            with lock:
                if np.isnan([total_loss_value]):
                    total_loss_value = 0
                shared_stats.append([episode_reward, total_loss_value, episode_steps])
            # FIXME logging stats  

            env.reset()
            setup_camera(env, config)
            state = get_screen(env, device, config)
            episode_reward = 0.0
            episode_steps = 0

        if current_ep is not None and current_ep >= max_episodes:
            break
    print(f"Metrics: ----------")
    print(f"Avg reward: {metrics.get_average_reward()}")
    print(f"Avg loss: {metrics.get_average_loss()}")
    print(f"Avg episode len: {metrics.get_average_episode_length()}")

    env.close()
