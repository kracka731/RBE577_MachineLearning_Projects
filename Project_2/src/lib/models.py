import torch
import torch.nn as nn
from typing import Dict, Any, List
import numpy as np
from tqdm import tqdm

from .physics import PushPhysics


class NNModel(nn.Module):
    """Base neural network architecture"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = [] 
        # Set up layers (6 hidden layers according to hidden_dims)
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)

        self.flatten = nn.Flatten()

        # Define an MLP with at least one hidden layer. 
        # Train the network using the given dataset.
        # Compute and visualize loss curves.
        # Compare the network's performance with the physics model.

        # models.py used in lib folder
        # custom.yaml in config
        
        # input data: x,y,theta,T,theta_push,d,D (7)
        # output data: x,y,theta (3)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits

    def loss(self, predictions, targets):
        # split into [x,y] and [theta]
        pos_pred, rot_pred = predictions[:,:1], predictions[:,2]
        pos_true, rot_true = targets[:,:1], targets[:,2]

        # Calculate MSE for position and rotation separately
        pos_loss = torch.mean((pos_pred - pos_true) ** 2)
        rot_loss = torch.mean((rot_pred - rot_true) ** 2)

        # Combine losses with weights (just 1 for now)
        total_loss = 1*pos_loss + 1*rot_loss
        return total_loss 

    def accuracy(self, validate_loader, pred):
        test_loss, correct = 0, 0
        for X, y in validate_loader:
            # test_loss += self.loss(predictions=pred, targets=y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        correct /= len(pred)
        return 100*correct


class NNPhysicsModel(NNModel):
    """Neural network with physics knowledge"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        physics: PushPhysics,
    ):
        super().__init__(input_dim + output_dim, output_dim, hidden_dims)
        self.physics = physics
        self.requires_grad = True

    # TODO: Implement forward function


class PushPlanner:
    """High-level push planning and training"""

    def __init__(
        self, model_config: Dict[str, Any], physics_sampling_config: Dict[str, Any]
    ):
        self.model_config = model_config
        self.physics_sampling_config = physics_sampling_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TODO: Initialize models
        physics_config = model_config["physics"]
        physics = PushPhysics.from_config(physics_config)

        net_config = self.model_config["network"]
        self.push_model = NNPhysicsModel(net_config["input_dim"], net_config["task_dim"], net_config["hidden_dims"], physics)
        
        # TODO: Move models to device
        self.push_model = self.push_model.to(self.device)

        # TODO: Setup optimizers
        learning_rate = self.model_config["optimizer"]["learning_rate"]
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.push_model.parameters(), lr=learning_rate)


    # TODO: Implement optimize_push function
    def optimize_push(self):
        pass

    # TODO: Implement plan_push function
    def plan_push(self):
        pass

    # TODO: Implement train_epoch function
    def train_epoch(self):
        pass


class PushNetFactory:
    """Factory for creating different types of push networks"""

    @staticmethod
    def create(config: Dict[str, Any]) -> nn.Module:
        network_config = config["network"]
        physics_config = config["physics"]
        model_type = network_config["type"]
        hidden_dims = network_config["hidden_dims"]

        if model_type == "NNModel":
            return NNModel(
                network_config["input_dim"], network_config["task_dim"], hidden_dims
            )
        elif model_type == "PhysicsModel":
            return PushPhysics.from_config(physics_config)
        else:
            physics = PushPhysics.from_config(physics_config)
            return NNPhysicsModel(
                network_config["input_dim"],
                network_config["task_dim"],
                hidden_dims,
                physics,
            )

    # TODO: Expand factory as needed
