import torch
import torch.nn as nn
from typing import Dict, Any, List
import numpy as np
from tqdm import tqdm
from helpers.config import load_config

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

    def accuracy(self, validate_loader, pred):
        test_loss, correct = 0, 0
        for X, y in validate_loader:
            # test_loss += self.loss(predictions=pred, targets=y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        correct /= len(pred)
        return 100*correct
    

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X, y = X.to(self.device)
            y_pred = self.model(X)
        return y_pred


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
    def forward(self, x):
        # phys_pred = self.physics.compute_motion(x)
        # x = torch.cat([x, phys_pred], dim=1)
        base_pred = super().forward(x)
        # print(f"type of p {type(phys_pred)} and base {type(base_pred)}")
        return base_pred #+ phys_pred


class PushPlanner:
    """High-level push planning and training"""

    def __init__(
        self, model_config: Dict[str, Any], physics_sampling_config: Dict[str, Any], override_model=None 
    ):
        self.model_config = model_config
        self.physics_sampling_config = physics_sampling_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TODO: Initialize models
        self.physics = PushPhysics.from_config(self.model_config['physics'])
        self.model = PushNetFactory.create(self.model_config, override_model)

        if type(self.model) != PushPhysics:
            # TODO: Move models to device
            self.model = self.model.to(self.device)

            # TODO: Setup optimizers
            learning_rate = self.model_config["optimizer"]["learning_rate"]
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

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

    # TODO: Implement optimize_push function
    def optimize_push(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def test(self, dataloader):
        self.model.eval()
        test_loss, correct = 0, 0
        y_actual = torch.zeros((1,3)).to(self.device)
        y_pred = torch.zeros((1,3)).to(self.device)

        # Make predictions with the model
        with torch.no_grad():
            # Move tensors to appropriate device
            # Do forward and calculate error
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)

                y_actual = torch.cat([y_actual, y], dim=0)
                y_pred = torch.cat([y_pred, pred], dim=0)

                test_loss += self.loss(predictions=pred, targets=y).item()

                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
        test_loss /= len(dataloader)
        correct /= len(dataloader.dataset)
        # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        # No need to save model states at this point in time. Validation is just for QOL.

        return test_loss, correct, y_actual, y_pred

    # TODO: Implement plan_push function
    def plan_push(self):
        pass

    def train_epoch(self, loaded_data, batch_size):
        self.model.train()        

        for batch, (X, y) in enumerate(loaded_data):
            # print("Batch: ",batch)
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(X) 
            loss = self.loss(pred, y)

            # Backpropagation
            loss.backward()
            self.optimize_push()

            # If desire to get a better idea of these values over time, uncomment
            # if batch % 7 == 0: # 21 batches, get 3 readings.
            #     loss, current = loss.item(), batch * batch_size + len(X) # .item() converts loss from tensor to float
            #     print(f"train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def phys_first(self, loaded_dataset):
        physics_predictions = self.physics.physics_pred(loaded_dataset)
        phys_pred_float = torch.zeros((len(physics_predictions), len(physics_predictions[0])) )

        # Iterate through prediction tensors and convert into floats
        # Probably not the most efficient method, but I don't notice the time
        for row in range(len(physics_predictions)): # for each row
            for col in range(len(physics_predictions[0])): # and each column in said row
                tensor = physics_predictions[row][col]

                phys_pred_float[row][col] = tensor.item()
        return phys_pred_float


class PushNetFactory:
    """Factory for creating different types of push networks"""

    @staticmethod
    def create(config: Dict[str, Any], override_model=None) -> nn.Module:
        network_config = config["network"]
        physics_config = config["physics"]
        model_type = network_config["type"]
        hidden_dims = network_config["hidden_dims"]
    
        match override_model:
            # If model not defined in command line arguments, use YAML-based config
            case None:
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
                
            case "nn":
                return NNModel(
                    network_config["input_dim"], network_config["task_dim"], hidden_dims
                )
            
            case "physics":
                return PushPhysics.from_config(physics_config)
            
            case "nn+physics":
                physics = PushPhysics.from_config(physics_config)
                return NNPhysicsModel(
                    network_config["input_dim"],
                    network_config["task_dim"],
                    hidden_dims,
                    physics,
                )

    # TODO: Expand factory as needed
