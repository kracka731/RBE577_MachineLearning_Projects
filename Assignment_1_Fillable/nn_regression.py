import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from helpers.loss import CustomLoss
from helpers.metrics import compute_mse, compute_position_error, compute_rotation_error
from datasets import prepare_dataset
from visualizer import plot_linear_trajectory
from sklearn.model_selection import train_test_split



def convert_to_tensor(data, device="cpu"):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float().to(device)
    return torch.tensor(data, dtype=torch.float32).to(device)

class MLP(nn.Module):
    def __init__(self, input_size=6, hidden_sizes=[128, 64, 32], output_size=6):
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
        layers = [3] # At least 1 hidden layer, more possible

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential( # Three hidden layers
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2],output_size)
        )

    def network(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def forward(self, x):
        return self.network(x)
    
    def divide_data(self, X, y):
        #70% training, 10% validation, 20% test
        # We just need to split the 80% X, into (7/8) and (1/8)
        X_train, X_validate, y_train, y_validate = train_test_split(
        X, y, test_size=1/8, random_state=42)
        return X_train, X_validate, y_train, y_validate

    def fit(
        self,
        X_train,
        y_train,
        lr=0.001,
        batch_size=32,
        epochs=100,
        device="gpu",
        ):

        X_train, X_validate, y_train, y_validate = self.divide_data(X_train, y_train)
        print("Device: ",device)
        # Convert to tensors
        X_train_tensor = convert_to_tensor(X_train, device)
        y_train_tensor = convert_to_tensor(y_train, device)
        X_validate_tensor = convert_to_tensor(X_validate, device)
        y_validate_tensor = convert_to_tensor(y_validate, device)


        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validate_dataset = TensorDataset(X_validate_tensor, y_validate_tensor)
        validate_loader = DataLoader(validate_dataset, batch_size=batch_size)


        # Initialize model, loss, and optimizer
        criterion = CustomLoss(position_weight=1.0, rotation_weight=1.0)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr) # not SGD, instead uses Adaptive Moment Estimation


        #### Your CODE STARTS HERE ####
        trainset_size = len(train_dataset)
        testset_size = len(validate_dataset)
        num_test_batches = len(validate_loader)

        for t in range(epochs):
            print(f"Epoch {t+1}\n--------------")
            
            # Training loop

            for batch, (X, y) in enumerate(train_loader):
                # Compute prediction and loss
                pred = self(X) 

                loss = criterion(predictions=pred, targets=y)

                # Backpropagation
                # View this video on back-propagation: https://www.youtube.com/watch?v=tIeHLnjs5U8
                # Massively helpful for understanding the math behind the scenes
                loss.backward() # deposits the gradients of the loss w.r.t. each parameter through backpropagation
                optimizer.step() # adjust parameters by the collected gradients collected in the backward pass
                optimizer.zero_grad() # reset gradients of model parameters. This prevents double-counting

                if batch % 100 == 0: # Show results over time
                    # print("train pred: ",pred)
                    loss, current = loss.item(), batch * batch_size + len(X)
                    print(f"loss: {loss:>7f} [{current:>5d}/{trainset_size:>5d}]")

            # Testing (evaluation) loop

            self.eval() # Set model to evaluation mode for batch normalization
            test_loss, correct = 0, 0

            with torch.no_grad(): # Ensures no gradients are computed during testing below
                for X, y in validate_loader:
                    pred = self(X) # Tensor size of 32, 6 long
                    # print("test pred: ",pred)
                    test_loss += criterion(predictions=pred, targets=y).item()
                    # print("test_loss: ",test_loss)
                    # print("y: ",y)
                    # x[32,6], y[32,6]
                    correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

            test_loss /= num_test_batches
            correct /= testset_size
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        #### Your CODE ENDS HERE ####
    
        return self

    def predict(self, X, device="cuda"): # Currently unused. Is there a way I can? Or a reason to? May be an alternative to testing loop portion. But the forced convert to tensor line gets in the way.
        X_tensor = convert_to_tensor(X, device)
        self.eval()
        with torch.no_grad():
            y_pred = self(X_tensor).cpu().numpy()
        return y_pred   

if __name__ == "__main__":

    # Load and prepare data
    X_train, X_test, y_train, y_test = prepare_dataset("data/ur10_dataset.csv")

    model = MLP().to("cuda")
    print("Model: \n",model) # For the world to see.

    # Train model
    model.fit(
        X_train.values,
        y_train.values,
        lr=0.001,
        batch_size=32,
        epochs=5,
        device="cuda",
    )

    plot_linear_trajectory(model)