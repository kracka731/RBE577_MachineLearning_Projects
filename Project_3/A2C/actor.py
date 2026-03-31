import torch
import torch.nn as nn
import numpy as np
import random

class Actor(nn.Module):
    """Policy network for discrete LunarLander actions."""

    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.action_dim = action_dim
        # TODO: Build the policy network
        # Hint: This module should transform a state vector into one score per action.
        self.nn = None  # Replace with your implementation
        layers = [] 
        # Set up layers (6 hidden layers according to hidden_dims)
        prev_dim = state_dim
        for dim in hidden_dim:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.layers = nn.Sequential(*layers)

        self.flatten = nn.Flatten()

        # Optimizer and other related setup handled in train.py

    def R(self, s, a):
        reward = 0
        # Shaping based on proximity to landing pad & stability 
        # Encourage small velocity, upright angle 

        # Penalty for using fuel when firing engines 
        if a != 0:
            reward -= 1

        # Reward for successful landing 
        if s[6] == 1 and s[7] == 1: 
            reward += 100

        # Penalty for crashing 
        if crash:
            reward -= 100

        return reward

    def evaluate_actions(self, states, actions):
        """Return chosen-action log probs and policy entropy."""
        # calculate the probabilities of the actor executing each action and then choose one
        # Called with a set of known taken actions and known lived states
        # Calculate specifically 

        # State space
        # x, y: horizontal and vertical position
        # x_hat, y_hat: horizontal and vertical velocities
        # theta: angle of the lander
        # theta_hat: angular velocity
        # c_L, c_R: contact indicators for left and right legs {0,1}
        # s = (x, y, x_hat, y_hat, theta, theta_hat, c_L, c_R)

        # Action space
        # 0: do nothing | 1: fire left orientation engine 
        # 2: fire main engine | 3: fire right orientation engine

        #TODO Fill your code
        # Forward pass
        logits = self.layers(states)  # Replace with your implementation
        # A logit is a bijective function that maps probabilities ([0,1])
        # to R((-ing, ing))

        # TODO: Convert the raw outputs into log-probabilities
        # Hint: The loss is written in terms of log probabilities rather than plain probabilities.
        # The log probablities of all possible actions
        log_action_probs = torch.log(logits)  # Replace with your implementation

        # Hint: You will need these when measuring how uncertain the policy is.
        # This portion uses what the model has learned to predict the likely best next action
        # Specifically considering the current state 
        action_probs = self(states)  # Replace with your implementation
        action_indices = np.array(actions, dtype=np.int32)

        # TODO: Mark which action was selected at each step
        # Hint: The provided `action` tensor contains indices, but you need a representation
        # that can isolate one action per row from the full action distribution.
        action_oh = torch.one_hot(action_indices) # Replace with your implementation


        # TODO: Extract the log-probability of each chosen action
        # Hint: Use the selected-action mask together with the full table of log probabilities.
        # log pi_theta(a|s)
        chosen_log_probs = torch.math.log(torch.reduce_sum(action_probs * action_oh))  # Replace with your implementation


        # TODO: Compute the entropy of the action distribution
        # Hint: Entropy should be larger when the policy is spread out and smaller when it is confident.
        entropy = None  # Replace with your implementation

        return chosen_log_probs, entropy
    
    def get_action(self, state, deterministic=False):
        # TODO: Run the policy on a single state - Forward pass
        logits = self.layers(state)  # Replace with your implementation


        # TODO: Return a greedy action when deterministic evaluation is requested
        if deterministic:
            # Choose the best action
            action = self(state)
            pass  # Replace with your implementation
        else: # stochastic, randomly choose an action 
            action = random.randrange(0, 4, 1) # choose a random action [0,1,2,3]

        # categorical function can give categorical distribution from softmax 
        action_dist = None # Replace with your implementation

        # TODO: Sample and return one action
        return action  # Replace with your implementation

