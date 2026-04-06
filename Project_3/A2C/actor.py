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

    # def R(self, s, a):
    #     reward = 0
    #     # Shaping based on proximity to landing pad & stability 
    #     # Encourage small velocity, upright angle 

    #     # Penalty for using fuel when firing engines 
    #     if a != 0:
    #         reward -= 1

    #     # Reward for successful landing 
    #     if s[6] == 1 and s[7] == 1: 
    #         reward += 100

    #     # Penalty for crashing 
    #     if crash:
    #         reward -= 100

    #     return reward

    def evaluate_actions(self, state, action):
        """Return chosen-action log probs and policy entropy."""
        # calculate the probabilities of the actor executing each action and then choose one
        #TODO Figure out if this should be for just one state and 4 actions, or for all known states and actions
        # Currently called with a set of known taken actions and known lived states
        # This is all to figure out the advantage of a certian action in a certian state
        # This is specifically the difference of the action value of a state-action pair with the state value of that state

        # state: current state
        # action: chosen action???? Then what's the point of the actor?
        # I guess this is not what chooses. Just thinks about the decision. Is that not what the critic does??

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
        entropy_coef = 1

        #TODO Fill your code
        # Forward pass
        action_logits = self.layers(state) 
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
        log_action_probs = torch.log(action_logits)

        # TODO may need to be altered to properly manage tensors (but is good according to practices online)
        # actions = torch.cat(actions, self.get_action(state[-1], False), 0) # choose an action and append it (non-deterministic) 
        # action_indices = np.array(action, dtype=np.int32)
        action_indice = torch.zeros([1,4])
        # [row][column]
        action_indice[0][action] = torch.tensor1 # assuming action is an integer from 0-3
        

        # TODO: Mark which action was selected at each step
        # Hint: The provided `action` tensor contains indices, but you need a representation
        # that can isolate one action per row from the full action distribution.
        # action_oh = torch.one_hot() # do torch.one_hot somehow
        action_oh = action_indice[0][action] = torch.tensor(1) # assuming action is an integer from 0-3

        # action_oh stands for action_one-hot

        # TODO: Extract the log-probability of each chosen action
        # Hint: Use the selected-action mask together with the full table of log probabilities.
        # chosen_log_probs = torch.math.log(torch.reduce_sum(action_probs * action_oh))  # Replace with your implementation
        chosen_log_prob = log_action_probs * action_oh

        # TODO: Compute the entropy of the action distribution
        # Hint: Entropy should be larger when the policy is spread out and smaller when it is confident.
        # An entropy bonus can be added to the actor's objectrage exploration (but not in this implementation?)
        # entropy = -torch.reduce_sum(chosen_log_prob * action_logits, entropy_coef) ive to encou
        entropy = action_logits.mean() # * entropy_coef
        return chosen_log_prob, entropy
    
    def get_action(self, state, deterministic=False):
        # TODO: Run the policy on a single state - Forward pass
        logits = self.layers(state) 

        # TODO: Return a greedy action when deterministic evaluation is requested
        if deterministic:
            with torch.no_grad(): # to not train during decision
                # Choose the best action
                action = self(state)
            pass  # Replace with your implementation
        else: # stochastic, randomly choose an action 
            action = random.randrange(0, 4, 1) # choose a random action [0,1,2,3]

        dist = torch.Categorical(logits=logits)

        # categorical function can give categorical distribution from softmax 
        action = dist.sample()
        # log_prob = dist.log_prob(action)

        # TODO: Sample and return one action
        return action  # Replace with your implementation

