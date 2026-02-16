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

    def __init__(self, learning_rate):
        self.lr = learning_rate

    def _initialize_parameters(self, input_dim, output_dim):
        """Initialize model weights and bias.

        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of output dimensions
        """
        # print(f"in/out features: {input_dim, output_dim}")
        self.weights = np.zeros((input_dim, output_dim)) # add row for the intercept/bias
        # print(f"weights shape: {np.shape(self.weights)}")

    def _compute_loss(self, y_pred, y_true):
        """Compute MSE loss between predictions and targets.

        Args:
            y_pred (np.ndarray): Model predictions
            y_true (np.ndarray): Ground truth values

        Returns:
            float: MSE loss value
        """
        return compute_mse(y_pred, y_true) 

    def _compute_gradients(self, X, y_true, y_pred):
        """Compute gradients for weights and bias.

        Args:
            X (np.ndarray): Input features
            y_true (np.ndarray): Ground truth values
            y_pred (np.ndarray): Model predictions

        Returns:
            tuple: Weight gradients and bias gradients
        """
        error = y_pred - y_true 
        weight_gradient = -2 * np.mean(X*error)
        bias_gradient = -2 * np.mean(error)
        return (weight_gradient, bias_gradient)

    def fit(self, X:np.ndarray, y:np.ndarray, batch_size=32, epochs=100):
        """Train model using mini-batch SGD.

        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features)
            y (np.ndarray): Target values of shape (n_samples, n_outputs)
            batch_size (int): Mini-batch size for SGD
            epochs (int): Number of training epochs
        """
        n_samples = np.size(X, axis=0)
        num_batches = int(np.ceil(n_samples/batch_size))

        for episode in range(epochs):
            # Reshuffle the data every episode to prevent overfitting
            data = np.hstack((X, y))
            np.random.shuffle(data)

            for batch in range(num_batches): # Examine one batch of data at a time 
                # Extract slices of the data based on batch number & batch size 
                i = batch*batch_size        # starting data num index 
                j = (batch+1)*batch_size    # ending data num index
                X_batch = data[i:j, :np.size(X, axis=1)]
                y_batch = data[i:j,  np.size(X, axis=1):]
                # print(f"dims of weights slice: {np.shape(self.weights)}, now T: {np.shape(self.weights.T)}, X batch; {np.shape(X_batch)}")
                y_pred = X_batch @ self.weights# (self.weights @ X_batch.T).T
                # print(f"y pred shape: {np.shape(y_pred)}")

                # Compute gradient & weights for this batch
                weight_grad, bias_grad = self._compute_gradients(X_batch, y_batch, y_pred)
                # self.weights[0] += self.lr * bias_grad # TODO: double check??
                self.weights += self.lr * weight_grad 
            y_pred = X @ self.weights
            # print(f"loss: {self._compute_loss(y_pred, y)}")



    def predict(self, X):
        """Make predictions for given input features.

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted values of shape (n_samples, n_outputs)
        """
        y = X @ self.weights # (self.weights @ X.T).T
        # print(f"shape of y_pred final: {np.shape(y)}")
        return y


if __name__ == "__main__":
    use_engineered_features = False 


    #############Your CODE STARTS HERE##############

    # Load data
    # X_train, X_test, y_train, y_test = prepare_dataset("data/ur10_dataset.csv")
    X_train, X_test, y_train, y_test = prepare_dataset("data/ur10_linear_dataset.csv")

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
    model._initialize_parameters(np.size(X_train, axis=1), np.size(y_train, axis=1))
    model.fit(X_train, y_train)
    # model.predict(X_test)

    #############Your CODE ENDS HERE##############

    plot_linear_trajectory(model, use_engineered_features=use_engineered_features)