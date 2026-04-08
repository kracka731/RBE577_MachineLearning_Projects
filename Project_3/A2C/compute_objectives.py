import torch


def compute_discounted_returns(rewards, gamma, bootstrap_value=None):
    """Compute the discounted return at every timestep """
    # Hint: Work backward from the end of the episode or rollout.
    # Gamma: discount factor from (0-1]

    discounted_returns = torch.tensor([0]) # this is the list of all returns
    # Called return instead of reward due to the causality assumption

    # Initialize the running return Gt = the summation of returns for a timestep t
    running_return = 0  
    

    # Accumulate discounted returns in reverse order
    # Hint: Each earlier timestep should include its own reward plus a discounted
    # contribution from what comes after it.
    # So this means, start from the end (episode termination)   
    t = len(rewards) - 1
    for reward in reversed(rewards):
        # print(f"rewards: {reward}")
        running_return += gamma**(t) * reward
        discounted_returns = torch.vstack([discounted_returns, torch.tensor(running_return)])
        t -= 1

    # Package the per-step returns into a single tensor
    # Hint: The training code expects one tensor containing all timesteps.
    discounted_returns = torch.flip(discounted_returns[1:], (0,))
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
    return return_batch - value_batch 
    # return normalize_advantage(return_batch - value_batch)

def normalize_advantage(advantage_batch):
    if advantage_batch.numel() <= 1:
        return advantage_batch
    return (
        advantage_batch - advantage_batch.mean()
    ) / (advantage_batch.std(unbiased=False) + 1e-8)
    # mean_adv = torch.mean(advantage_batch)
    # std_adv = torch.std(advantage_batch) + 1e-8
    # return (advantage_batch - mean_adv) / std_adv


def compute_actor_loss(chosen_log_probs, advantage_batch, grad_bounds=None):
    """Compute policy loss through REINFORCE: derivative of the objective 
    function = grad(J(theta))"""
    # First term sum(grad(log(policy)) 
    # Indicates direction to increase probability of actions taken by the policy
    # aka: how to adjust parameters (theta) to make observed sequence of actions

    # Second term Gt or At is advantage_batch 
    # Determines how strongly to reinforce the direction taken by policy gradient
    # advantage_batch = normalize_advantage(advantage_batch)
    grad = torch.mean(-chosen_log_probs * advantage_batch)

    # grad = torch.clamp(grad, min=-grad_bounds, max=grad_bounds)
    return grad

    # actor_loss = -(chosen_log_probs * advantage_batch.detach()).mean()
    # return actor_loss

def compute_critic_loss(return_batch, value_batch, value_loss_coeff):
    """Compute the MSE of the advantage"""
    # Compute the value-function loss
    advantage = compute_advantage(return_batch, value_batch)
    return torch.mean(advantage**2)*value_loss_coeff 
