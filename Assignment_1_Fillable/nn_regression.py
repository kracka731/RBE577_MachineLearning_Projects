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
        layers = [] # At least 1 hidden layer, more possible

        #TODO FILLOUT

        torch.cuda.init()

        self.weights = np.zeros(input_size)
        self.bias = 0

        self.learning_rate = 0.01 # test value. Can be altered
        # self.n_iterations = 1000 # Number of epochs. Don't do to much to prevent overfitting.
        self.weights = None # What the model will learn to adapt
        self.bias = None # Controls the activation threshold in the activation function

    def forward(self, x):
        return self.network(x)

    def _step_function(self, x, threshold: int = 0):
        # where x is greater than the threshold, yield 1
        # otherwise, yield 0
        # 0 is just a standard threshold value for this default activation function
        # bias will alter this in an abstracted fashion
        return np.where(x > threshold, 1, 0)

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
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training loop

        #### Your CODE STARTS HERE ####

        # Adapted from this website: https://www.freecodecamp.org/news/build-a-multilayer-perceptron-with-examples-and-python-code/
        # MLP needs at least one hidden layer.
        # Will initially design with one layer 

        n_samples, n_features = X_train.shape

        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(epochs):
            for i in range(n_samples):
                # compute weighted sum (ws)
                ws = np.dot(X_train[i], self.weights) + self.bias
                
                # apply the activation function
                y_pred = self._step_function(ws)

                self.weights += self.learning_rate * (y_train[i] - y_pred) * X_train[i]
                self.bias += self.learning_rate * (y_train[i] - y_pred)
        
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