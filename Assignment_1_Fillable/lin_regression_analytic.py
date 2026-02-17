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
        # Data parsed as such in datasets.py
        # X (number of samples, [th1, th2, th3, th4, th5, th6])
        # y (number of samples,  [x,y,z,rx,ry,rz])

        # Add column to represent the intercept/bias
        n_samples, n_inputs = np.shape(X)
        X = np.hstack((np.ones((n_samples, 1)), X))
        
        # We will be using the Ordinary Least Squares method
        XT = np.transpose(X)
        XTX_inv = np.linalg.inv(XT @ X) # The matrix of feature cross products

        # Find the estimated coefficient vector and label as weights
        self.weights = XTX_inv@XT@y

        # If the determinant of a matrix is 0, it is singular
        n_inputs, n_outputs = np.shape(self.weights)
        if n_inputs == n_outputs: 
            return np.linalg.det(self.weights) != 0
        else:
            return True

    def predict(self, X):
        """Make predictions for given input features.

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted values of shape (n_samples, n_outputs)
        """

        # print("weights: ",self.weights,"\n")
        # print("X: ",X,"\n")
        # Add column to represent the intercept/bias
        n_samples, n_inputs = np.shape(X)
        X = np.hstack((np.ones((n_samples, 1)), X))

        y_pred = X @ self.weights

        return y_pred

if __name__ == "__main__":

    #############Your CODE STARTS HERE##############
    use_engineered_features = True 

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
    success = model.fit(X_train, y_train)

    if success:
        y_pred = model.predict(X_test)



    #############Your CODE ENDS HERE##############

    plot_linear_trajectory(model, use_engineered_features=use_engineered_features)