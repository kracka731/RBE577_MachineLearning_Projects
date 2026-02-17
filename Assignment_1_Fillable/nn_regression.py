import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from helpers.loss import CustomLoss
from helpers.metrics import compute_mse, compute_position_error, compute_rotation_error
from datasets import prepare_dataset
from visualizer import plot_linear_trajectory


def convert_to_tensor(data, device="cpu"):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float().to(device)
    return torch.tensor(data, dtype=torch.float32).to(device)

class MLP(nn.Module):
    def __init__(self, input_size=6, hidden_sizes=[128, 64], output_size=6):
        """
        Initialize a Multi-Layer Perceptron (MLP) neural network.

        Args:
            input_size (int, optional): The number of input features. 
            hidden_sizes (list, optional): A list of integers representing the number of neurons 
                in each hidden layer. 
            output_size (int, optional): The number of output neurons. 

        Description:
            Constructs an MLP with the specified architecture. The network consists of:
            - An input layer that maps from input_size to the first hidden layer
            - Hidden layers with ReLU activation functions between them
            - An output layer with no activation function
            
            All layers are stored in a Sequential container accessible via self.network.
        """
        super(MLP, self).__init__()
        layers = [2] # At least 1 hidden layer, more possible

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential( # Two hidden layers
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1],output_size)
        )

    def network(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def forward(self, x):
        return self.network(x)

    def fit(
        self,
        X_train,
        y_train,
        lr=0.001,
        batch_size=32,
        epochs=100,
        device="gpu",
        ):

        # Convert to tensors
        X_train_tensor = convert_to_tensor(X_train, device)
        y_train_tensor = convert_to_tensor(y_train, device)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize model, loss, and optimizer
        criterion = CustomLoss(position_weight=1.0, rotation_weight=1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) # not SGD, instead uses Adaptive Moment Estimation

        # Training loop

        #### Your CODE STARTS HERE ####
        size = len(train_loader)

        for batch, (X, y) in enumerate(train_loader):
            # Compute prediction and loss
            pred = model(X) #need to implement this correctly
            loss = criterion(predictions=pred, targets=y)

            # Backpropagation
            # View this video on back-propagation: https://www.youtube.com/watch?v=tIeHLnjs5U8
            # Massively helpful for understanding the math behind the scenes
            loss.backward() # deposits the gradients of the loss w.r.t. each parameter through backpropagation
            optimizer.step() # adjust parameters by the collected gradients collected in the backward pass
            optimizer.zero_grad() # reset gradients of model parameters. This prevents double-counting

            if batch % 100 == 0: # Show results over time
                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


        #### Your CODE ENDS HERE ####
    
        return model

    def predict(self, X, device="cuda"):
        X_tensor = convert_to_tensor(X, device)
        self.eval()
        with torch.no_grad():
            y_pred = self(X_tensor).cpu().numpy()
        return y_pred   

if __name__ == "__main__":

    # Load and prepare data
    X_train, X_test, y_train, y_test = prepare_dataset("data/ur10_dataset.csv")

    model = MLP().to("cuda")
    print("Model: \n",model)

    # Train model
    model.fit(
        X_train.values,
        y_train.values,
        lr=0.001,
        batch_size=32,
        epochs=100,
        device="cuda",
    )

    plot_linear_trajectory(model)