import torch


def compute_discounted_returns(rewards, gamma, bootstrap_value=None):
    # TODO: Compute the discounted return at every timestep
    # Hint: Work backward from the end of the episode or rollout.
    # Gamma: discount factor from (0-1]

    discounted_returns = [] # G in the math, this is the list of all returns?
    # Called return instead of reward due to the causality assumption

    # TODO: Initialize the running return
    running_return = None  # Replace with your implementation
    

    # TODO: Accumulate discounted returns in reverse order
    # Hint: Each earlier timestep should include its own reward plus a discounted
    # contribution from what comes after it.
    # So this means, start from the goal?
    
    for reward in reversed(rewards):
        pass  # Replace with your implementation


    # TODO: Package the per-step returns into a single tensor
    # Hint: The training code expects one tensor containing all timesteps.
    # Look at last project?

    return None  # Replace with your implementation

def compute_advantage(return_batch, value_batch):
    # TODO: Compute the advantage estimate
    # Hint: This quantity should capture how much better or worse the observed return
    # was compared with the critic's prediction.
    return None  # Replace with your implementation

def normalize_advantage(advantage_batch):
    if advantage_batch.numel() <= 1:
        return advantage_batch
    return (
        advantage_batch - advantage_batch.mean()
    ) / (advantage_batch.std(unbiased=False) + 1e-8)


def compute_actor_loss(chosen_log_probs, advantage_batch):
    # TODO: Compute the policy loss

    # J(theta)

    return None  # Replace with your implementation

def compute_critic_loss(return_batch, value_batch):
    # TODO: Compute the value-function loss
    return None  # Replace with your implementation
