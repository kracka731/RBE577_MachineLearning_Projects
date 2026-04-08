import torch
import torch.nn as nn
import numpy as np
import random
from torch.distributions import Categorical

class Actor(nn.Module):
    """Policy network for discrete LunarLander actions."""

    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        # self.action_dim = action_dim
        # TODO: Build the policy network
        # Hint: This module should transform a state vector into one score per action.
        # self.nn = None  # Replace with your implementation
        layers = [] 

        # layer count is actually not described in the config file, so assuming here
        layers.extend([nn.Linear(state_dim, hidden_dim), nn.ReLU()])
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, action_dim))

        self.layers = nn.Sequential(*layers)

        self.flatten = nn.Flatten()
        # Optimizer and other related setup handled in train.py

    def forward(self, state):
        # flat_state=self.flatten(state)
        # logits = self.layers(flat_state)
        logits = self.layers(state)
        return torch.softmax(logits, dim = -1)

    def evaluate_actions(self, states, actions):
        """Return chosen-action log probs and policy entropy."""
        # states: tensor array of lived states
        # actions: tensor array of chosen actions

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

        # Forward pass
        action_logits = self(states)
        # A logit is a bijective function that maps probabilities ([0,1])
        # Can be articulated is pi_theta(a_i, s_i) in math
        # Also refered to as action logits
        # which, thus means these are action probabilities at the current state

        # Hint: You will need these when measuring how uncertain the policy is.
        # This portion uses what the model has learned to predict the likely best next action
        # Specifically considering the current state 
        dist = Categorical(action_logits)

        # Convert the raw outputs into log-probabilities
        # Hint: The loss is written in terms of log probabilities rather than plain probabilities.
        # The log probablities of all possible actions
        # log pi_theta(a|s)
        actions = actions.view(-1)
        chosen_log_probs = dist.log_prob(actions)

        # TODO: Compute the entropy of the action distribution
        # Hint: Entropy should be larger when the policy is spread out and smaller when it is confident.
        # An entropy bonus can be added to the actor's objectrage exploration (but not in this implementation?)
        entropy = dist.entropy()

        return chosen_log_probs, entropy
    
    def get_action(self, state, deterministic):
        # Run the policy on a single state - Forward pass
        with torch.no_grad(): # In order to not include the gradient function to save time and computation
            logits = self(state) 

        # Return a greedy action when deterministic evaluation is requested
        if deterministic:
            # Choose the best action
            # consider logits and choose one with the highest probability to be chosen action
            action = int(torch.argmax(logits).item())

            # print(f"LOGITS:    {logits}")
            # print(f"ACTION:    {action}")

        else: # stochastic, randomly choose an action 
            dist = Categorical(logits=logits)
            action = int(dist.sample())
            
        # categorical function can give categorical distribution from softmax 
        
        # Return one action
        return action

