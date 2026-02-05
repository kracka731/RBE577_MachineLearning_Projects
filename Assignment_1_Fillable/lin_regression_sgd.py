import numpy as np
from helpers.metrics import compute_mse, compute_position_error, compute_rotation_error
from datasets import prepare_dataset
from visualizer import plot_linear_trajectory


class SGDLinearRegression:
    """Linear regression implementation using stochastic gradient descent optimization.

    Attributes:
        weights (np.ndarray): Model weights
        bias (np.ndarray): Model bias
        lr (float): Learning rate for gradient descent

    Methods:
        fit(X, y): Train model using mini-batch SGD
        predict(X): Make predictions on new data

    Example:
        >>> model = SGDLinearRegression(learning_rate=0.01)
        >>> model.fit(X_train, y_train, batch_size=32, epochs=100)
        >>> y_pred = model.predict(X_test)
    """

    def _initialize_parameters(self, input_dim, output_dim):
        """Initialize model weights and bias.

        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of output dimensions
        """
        pass

    def _compute_loss(self, y_pred, y_true):
        """Compute MSE loss between predictions and targets.

        Args:
            y_pred (np.ndarray): Model predictions
            y_true (np.ndarray): Ground truth values

        Returns:
            float: MSE loss value
        """
        pass

    def _compute_gradients(self, X, y_true, y_pred):
        """Compute gradients for weights and bias.

        Args:
            X (np.ndarray): Input features
            y_true (np.ndarray): Ground truth values
            y_pred (np.ndarray): Model predictions

        Returns:
            tuple: Weight gradients and bias gradients
        """
        pass

    def fit(self, X, y, batch_size=32, epochs=100):
        """Train model using mini-batch SGD.

        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features)
            y (np.ndarray): Target values of shape (n_samples, n_outputs)
            batch_size (int): Mini-batch size for SGD
            epochs (int): Number of training epochs
        """
        pass

    def predict(self, X):
        """Make predictions for given input features.

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted values of shape (n_samples, n_outputs)
        """
        pass


if __name__ == "__main__":
    use_engineered_features = True 


    #############Your CODE STARTS HERE##############

    # Load data
    X_train, X_test, y_train, y_test = prepare_dataset("data/ur10_dataset.csv")

    # Convert to numpy
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values

    if use_engineered_features:
        from feature_engineering import engineer_features
        X_train = engineer_features(X_train)
        X_test = engineer_features(X_test) 

    # Train model
    model = SGDLinearRegression(learning_rate=0.01)
    model.fit(X_train, y_train)

    #############Your CODE ENDS HERE##############

    plot_linear_trajectory(model, use_engineered_features=use_engineered_features)