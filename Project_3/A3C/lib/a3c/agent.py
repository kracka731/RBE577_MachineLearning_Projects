import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

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
        env.reset()  # Replace with your implementation
        setup_camera(env, config)
        state = get_screen(env, device, config)
        #FIXME: probably missing things here

        log_probs = torch.empty((1, action_dim)).to(device)
        values = []
        rewards = []
        entropies = torch.empty((1, action_dim)).to(device)
        done = False

        for _ in range(t_max):
            # TODO: Run the local network to get the current policy output and value estimate
            # Hint: The model returns the actor output and critic value; then use the
            # model helper to turn the actor output into a distribution.
            action_loc, value = local_net(state)  # Replace with your implementation
            dist = local_net.get_action_distribution(action_loc)  # Replace with your implementation

            # TODO: Sample an action and compute the policy terms needed later
            action = dist.rsample().flatten()  
            log_prob = dist.log_prob(action)  # Replace with your implementation
            entropy = -log_prob  # FIXME slightly inaccurate since dist.entropy doesnt work
            action_np = None  # Replace with your implementation

            # TODO: Step the environment and preprocess the next observation
            observation, reward, done, debug = env.step(action)
            next_state = get_screen(env, device, config)  # Replace with your implementation

            # TODO: Save the rollout information needed for the loss computation
            # Hint: Store the policy terms, value estimates, and rewards one step at a time.
            log_probs = torch.vstack([log_probs, log_prob])
            values.append(value)
            rewards.append(reward)
            entropies = torch.vstack([entropies, entropy])

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
        log_prob_batch = log_probs
        value_batch = torch.tensor(values).squeeze()
        entropy_batch = entropies 
        
        advantage_batch = compute_advantage(return_batch, value_batch.detach()).to(device) 
        actor_loss = compute_actor_loss(log_prob_batch, advantage_batch.detach(), entropy_batch, entropy_coef) 
        critic_loss = compute_critic_loss(return_batch.detach(), value_batch) 
        total_loss = actor_loss + value_loss_coef*critic_loss  

        # FIXME: i didnt write these next 4 lines but what are they???
        actor_loss_value = actor_loss.item()
        critic_loss_value = critic_loss.item()
        total_loss_value = total_loss.item()
        policy_std_value = float(F.softplus(local_net.sigma).mean().item())

        # TODO: Backpropagate through the local worker model and clip gradients
        # Hint: Gradients are still computed on the worker's local network first.
        # FIXME ???
        optimizer.zero_grad()
        total_loss.backward()
        for local_param, global_param in zip(local_net.parameters(), global_net.parameters()):
            global_param._grad = local_param.grad
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
            # FIXME ????  

            env.reset()
            setup_camera(env, config)
            state = get_screen(env, device, config)
            episode_reward = 0.0
            episode_steps = 0

        if current_ep is not None and current_ep >= max_episodes:
            break

    env.close()
