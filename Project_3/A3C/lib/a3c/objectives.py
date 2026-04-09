import torch
import torch.nn.functional as F

def compute_bootstrapped_returns(rewards, gamma, bootstrap_value):
    """Compute discounted returns with a bootstrap value at the rollout boundary."""
    # TODO: Initialize the running return from the rollout boundary
    # Hint: If the rollout ended before the episode terminated, this starting
    # value should carry the critic's estimate of what comes next.
    running_return = 0  # Replace with your implementation
    returns = []

    # TODO: Accumulate discounted returns backward through the rollout
    rewards.append(bootstrap_value)
    t = len(rewards) - 1
    for reward in reversed(rewards):
        running_return += gamma**(t) * reward
        returns.append(running_return)  
        t -= 1

    # TODO: Package the per-timestep returns into one tensor
    returns = torch.tensor(returns)
    returns = torch.flip(returns, (0,))
    return returns  # Replace with your implementation

def normalize_advantage(advantage_batch):
    if advantage_batch.numel() <= 1:
        return advantage_batch
    mean_adv = torch.mean(advantage_batch)
    std_adv = torch.std(advantage_batch) + 1e-8
    return (advantage_batch - mean_adv) / std_adv

def compute_advantage(return_batch, value_batch):
    """Plain actor-critic advantage."""
    # TODO: Compute how much better or worse the observed return was than
    # the critic's prediction at each timestep.
    adv = return_batch - value_batch
    return normalize_advantage(adv)  

def compute_actor_loss(log_prob_batch, advantage_batch, entropy_batch, entropy_coef):
    """Policy loss with an entropy bonus for exploration."""
    # TODO: Compute the policy-gradient term for the actor
    policy_loss = log_prob_batch * advantage_batch.reshape((len(advantage_batch), 1))
    entropy_bonus = entropy_coef * entropy_batch   
    loss = -policy_loss + entropy_bonus
    # FIXME: put -loss ? though that would be for minimizing reward ?
    return loss.mean()  # Replace with your implementation

def compute_critic_loss(return_batch, value_batch):
    """Mean-squared value regression loss."""
    # TODO: Compute the critic regression loss
    advantage = compute_advantage(return_batch, value_batch) # for MSE, return is input, value is target
    return torch.mean(torch.pow(advantage, 2))  # Replace with your implementation

