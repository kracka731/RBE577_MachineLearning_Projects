import numpy as np
from helpers.metrics import compute_mse, compute_position_error, compute_rotation_error
from datasets import prepare_dataset
from visualizer import plot_linear_trajectory


class AnalyticalLinearRegression:
    """Linear regression using closed-form analytical solution.

    Methods:
        fit(X, y): Compute weights using normal equation
        predict(X): Make predictions using learned weights

    Example:
        >>> model = AnalyticalLinearRegression()
        >>> success = model.fit(X_train, y_train)
        >>> if success:
        >>>     y_pred = model.predict(X_test)
    """

    def fit(self, X, y):
        """Compute weights using normal equation.

        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features)
            y (np.ndarray): Target values of shape (n_samples, n_outputs)

        Returns:
            bool: True if successful, False if matrix is singular
        """
        XT = np.transpose(X)
        XT_inv = np.linalg.inv(XT*X)
        weights = XT_inv*XT*y

        # If the determinant of a matrix is 0, it is singular
        if np.linalg.det(weights) == 0: 
            return False
        else:
            return True

    def predict(self, X):
        """Make predictions for given input features.

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted values of shape (n_samples, n_outputs)
        """
        pass


if __name__ == "__main__":

    #############Your CODE STARTS HERE##############
    use_engineered_features = False 

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
    model = AnalyticalLinearRegression()
    model.fit(X_train, y_train)

    #############Your CODE ENDS HERE##############

    plot_linear_trajectory(model, use_engineered_features=use_engineered_features)