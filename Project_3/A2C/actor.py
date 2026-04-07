import torch
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
        # calculate the probabilities of the actor executing each action and then choose one
        #TODO Figure out if this should be for just one state and 4 actions, or for all known states and actions
        # Currently called with a set of known taken actions and known lived states
        # This is all to figure out the advantage of a certian action in a certian state
        # This is specifically the difference of the action value of a state-action pair with the state value of that state

        # state: tensor array of lived states
        # action: tensor array of chosen actions

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

        # N = len(states) # Number of sampled experiences
        # entropy_coef = 1

        #TODO Fill your code
        # Forward pass
        with torch.no_grad(): # In order to not include the gradient function to save time and computation
            action_logits = self(states) 
            # action_logits = self.forward(states)
        # print(f"action_logits: {action_logits}")
        # print(f"action logits: {action_logits}")

        # A logit is a bijective function that maps probabilities ([0,1])
        # Can be articulated is pi_theta(a_i, s_i) in math
        # Also refered to as action logits
        # which, thus means these are action probabilities at the current state

        # Hint: You will need these when measuring how uncertain the policy is.
        # This portion uses what the model has learned to predict the likely best next action
        # Specifically considering the current state 
        # action_probs = self(state)

        # TODO: Convert the raw outputs into log-probabilities
        # Hint: The loss is written in terms of log probabilities rather than plain probabilities.
        # The log probablities of all possible actions
        # log pi_theta(a|s)
        log_action_probs = torch.log1p(action_logits)

        # 
        action_oh = torch.zeros([len(actions),4])
        # [row][column]
        

        # Mark which action was selected at each step
        # Hint: The provided `action` tensor contains indices, but you need a representation
        # that can isolate one action per row from the full action distribution.
        n=0
        for action in actions:
            action_oh[n][action] = torch.tensor(1.) # assuming action is an integer from 0-3
            n+=1

        # print(f"action_oh = {action_oh}")


        # action_oh stands for action_one-hot

        # Extract the log-probability of each chosen action
        # Hint: Use the selected-action mask together with the full table of log probabilities.
        # chosen_log_probs = torch.math.log(torch.reduce_sum(action_probs * action_oh))  # Replace with your implementation
        chosen_log_prob = -torch.sum(log_action_probs * action_oh, 1)
        # print(f"in actor chosen_log_prob: {chosen_log_prob}")

        # TODO: Compute the entropy of the action distribution
        # Hint: Entropy should be larger when the policy is spread out and smaller when it is confident.
        # An entropy bonus can be added to the actor's objectrage exploration (but not in this implementation?)
        entropy = action_logits.mean() # * entropy_coef
        # print(f"chosen_log_prob: {chosen_log_prob}")

        return chosen_log_prob, entropy
    
    def get_action(self, state, deterministic):
        # Run the policy on a single state - Forward pass
        # print(f"gat_action state: {state}")
        with torch.no_grad(): # In order to not include the gradient function to save time and computation
            logits = self(state) 
            # print(f"get_action logits: {logits}")

            # Return a greedy action when deterministic evaluation is requested
            if deterministic != False:
                # print("running deterministic")
                # Choose the best action
                # consider logits and choose one with the highest probability to be chosen action
                dist = Categorical(logits=logits)
                # print(f"dist: {dist}")
                action = dist.sample()

            elif deterministic == False: # stochastic, randomly choose an action 
                action = random.randrange(0, 4, 1) # choose a random action [0,1,2,3]
            
            else:
                raise ValueError(f"Invalid value of determinisitic: {deterministic}")

        # categorical function can give categorical distribution from softmax 
        # action = dist.sample()
        # log_prob = dist.log_prob(action)

        # Return one action
        return int(action)  # Replace with your implementation

