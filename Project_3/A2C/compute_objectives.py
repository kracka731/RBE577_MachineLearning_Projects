import torch


def compute_discounted_returns(rewards, gamma, bootstrap_value=None):
    """Compute the discounted return at every timestep """
    # Hint: Work backward from the end of the episode or rollout.
    # Gamma: discount factor from (0-1]

    discounted_returns = torch.empty((0,)) # this is the list of all returns
    # Called return instead of reward due to the causality assumption

    # Initialize the running return Gt = the summation of returns for a timestep t
    running_return = 0  

    # Accumulate discounted returns in reverse order
    # Hint: Each earlier timestep should include its own reward plus a discounted
    # contribution from what comes after it.
    # So this means, start from the end (episode termination)   
    t = len(rewards) - 1
    for reward in reversed(rewards):
        running_return += gamma**(t) * reward
        discounted_returns = torch.cat((discounted_returns, running_return))
        t -= 1

    # Package the per-step returns into a single tensor
    # Hint: The training code expects one tensor containing all timesteps.
    discounted_returns = torch.flip(discounted_returns, 0)

    return discounted_returns  

def compute_advantage(return_batch, value_batch):
    """A2C: compute difference in observed return vs critic's prediction"""
    # Compute the advantage estimate
    # Hint: This quantity should capture how much better or worse the observed return
    # was compared with the critic's prediction.
    return normalize_advantage(return_batch - value_batch)  

def normalize_advantage(advantage_batch):
    if advantage_batch.numel() <= 1:
        return advantage_batch
    return (
        advantage_batch - advantage_batch.mean()
    ) / (advantage_batch.std(unbiased=False) + 1e-8)


def compute_actor_loss(chosen_log_probs, advantage_batch):
    """Compute policy loss through REINFORCE: derivative of the objective 
    function = grad(J(theta))"""
    # First term sum(grad(log(policy)) 
    # Indicates direction to increase probability of actions taken by the policy
    # aka: how to adjust parameters (theta) to make observed sequence of actions

    # Second term Gt or At is advantage_batch 
    # Determines how strongly to reinforce the direction taken by policy gradient

    return torch.sum(chosen_log_probs * advantage_batch)

def compute_critic_loss(return_batch, value_batch):
    """Compute the MSE of the advantage"""
    # Compute the value-function loss
    advantage = compute_advantage(return_batch, value_batch)
    return torch.mean(advantage**2) 
