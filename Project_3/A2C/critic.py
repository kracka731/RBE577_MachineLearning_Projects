import torch.nn as nn
import torch


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()

        super().__init__()
        # self.action_dim = action_dim
        # TODO: Build the policy network
        # Hint: This module should transform a state vector into one score per action.
        # self.nn = None  # Replace with your implementation
        layers = [] 

        # layer count is actually not described in the config file, so assuming here
        layers.extend([nn.Linear(state_dim, hidden_dim), nn.ReLU()])
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))

        self.layers = nn.Sequential(*layers)

        self.flatten = nn.Flatten()


        # TODO: Build the value network
        # Hint: This network should take a state and return a single scalar estimate
        # describing how good that state is under the current policy.
        # self.nn = None  # Replace with your implementation

    def forward(self, state):
        # TODO: Predict the value of the input state
        # Hint: The forward pass should delegate to the network defined above.

        x = self.layers(state)
        return x  
        # return torch.softmax(x, dim = -1)
