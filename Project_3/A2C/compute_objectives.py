import torch
import numpy as np


def compute_discounted_returns(rewards, gamma, bootstrap_value=None):
    """Compute the discounted return at every timestep """
    # Hint: Work backward from the end of the episode or rollout.
    # Gamma: discount factor from (0-1]
    discounted_returns_list = []

    # discounted_returns = torch.tensor([0]) # this is the list of all returns
    # discounted_returns = np.zeros_like(rewards, dtype=np.float32)
    # Called return instead of reward due to the causality assumption
    
    # Initialize the running return Gt = the summation of returns for a timestep t
    running_return = 0  
    

    # Accumulate discounted returns in reverse order
    # Hint: Each earlier timestep should include its own reward plus a discounted
    # contribution from what comes after it.
    # So this means, start from the end (episode termination)   
    # t = len(rewards) - 1
    # for reward in reversed(rewards):
    #     # print(f"rewards: {reward}")
    #     # running_return += gamma**(t) * reward
    #     running_return = reward + gamma * running_return
    #     discounted_returns_list.append(running_return)
    #     # discounted_returns = torch.vstack([discounted_returns, torch.tensor(running_return)])
    #     t -= 1

    # for t in reversed(range(len(rewards))):
    #     running_return = rewards[t] + gamma*running_return
    #     discounted_returns[t] = running_return

    for r in reversed(rewards):
        running_return = r + gamma * running_return
        discounted_returns_list.insert(0, running_return) # prepend to keep order
    
    # discounted_returns = torch.tensor(np.stack(discounted_returns_list))
    discounted_returns = torch.tensor(discounted_returns_list)

    #normalize
    discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-9)

    # Package the per-step returns into a single tensor
    # Hint: The training code expects one tensor containing all timesteps.
    # discounted_returns = torch.flip(discounted_returns, (0,))
    # discounted_returns = torch.sum(discounted_returns)
    # print(f"Discounted returns: {discounted_returns}")

    return discounted_returns

    # advantages = torch.zeros_like(rewards)
    # last_advantage = 0.0
    # n_steps = len(rewards)

    # for t in reversed(range(n_steps)):

def compute_advantage(return_batch, value_batch):
    """A2C: compute difference in observed return vs critic's prediction"""
    # Compute the advantage estimate
    # Hint: This quantity should capture how much better or worse the observed return
    # was compared with the critic's prediction.
    # return normalize_advantage(return_batch - value_batch)  
    advantages = return_batch - value_batch
    # return (advantages - advantages.mean()) / (advantages.std() + 1e-9)
    return normalize_advantage(advantages)
    # return advantages

def compute_advantage_gae(return_batch, value_batch, next_value_batch, episode_terminations, config):
    advantages = torch.zeros_like(return_batch)
    last_advantage = 0.0
    n_steps = len(return_batch)
    lambda_gae = 0.9

    for t in reversed(range(n_steps)):
        mask = 1.0 - episode_terminations[t]
        delta = return_batch[t] + config['gamma'] * next_value_batch[t] * mask - value_batch[t]
        advantages[t] = delta + config['gamma'] * lambda_gae * last_advantage * mask
        last_advantage = advantages[t]
    
    return advantages

def normalize_advantage(advantage_batch):
    if advantage_batch.numel() <= 1:
        return advantage_batch
    return (
        advantage_batch - advantage_batch.mean()
    ) / (advantage_batch.std(unbiased=False) + 1e-8)
    # return (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-9)
    # mean_adv = torch.mean(advantage_batch)
    # std_adv = torch.std(advantage_batch) + 1e-8
    # return (advantage_batch - mean_adv) / std_adv


def compute_actor_loss(chosen_log_probs, reward_batch, grad_bounds=None):
    """Compute policy loss through REINFORCE: derivative of the objective 
    function = grad(J(theta))"""
    # First term sum(grad(log(policy)) 
    # Indicates direction to increase probability of actions taken by the policy
    # aka: how to adjust parameters (theta) to make observed sequence of actions

    # Second term Gt or At is advantage_batch 
    # Determines how strongly to reinforce the direction taken by policy gradient
    # advantage_batch = normalize_advantage(advantage_batch)
    
    # grad = -(chosen_log_probs * reward_batch).sum()
    # # grad = torch.clamp(grad, min=-grad_bounds, max=grad_bounds)
    # return grad

    # actor_loss = -(chosen_log_probs * advantage_batch.detach()).mean()
    # return actor_loss
    # normalize_advantage(advantage_batch)

    actor_loss = -(chosen_log_probs * reward_batch.detach()).sum()
    return actor_loss

    # loss = []
    # for log_prob, Gt in zip(chosen_log_probs, reward_batch):
    #     loss.append(-log_prob*Gt)
    
    # loss = torch.stack(loss).sum()
    
    # return loss

def compute_critic_loss(return_batch, value_batch, value_loss_coeff):
    """Compute the MSE of the advantage"""
    # Compute the value-function loss
    # advantage = compute_advantage(return_batch, value_batch)
    # print(f"advantage: {advantage}")
    # advantage = normalize_advantage(advantage)
    # return torch.mean(advantage**2)#*value_loss_coeff 
    # print(f"size1: {value_batch.type()}")
    # print(f"size2: {return_batch.detach().type()}")

    critic_loss = torch.nn.functional.mse_loss(value_batch, return_batch.detach())
    # print(f"size3: {critic_loss.type()}")

    return critic_loss
