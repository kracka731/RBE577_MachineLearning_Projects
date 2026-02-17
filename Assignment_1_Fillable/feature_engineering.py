import numpy as np


def engineer_features(angles):
    """Engineer features for robot kinematics based on forward kinematics equations.

    Creates features from joint angles that better capture the nonlinear relationships in robot forward kinematics.

    Args:
        angles (np.ndarray): Input joint angles array of shape (n_samples, 6)

    Returns:
        np.ndarray: Engineered features array of shape (n_samples, 42)

    Example:
        >>> angles = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        >>> features = engineer_features(angles)
        >>> print(features.shape)
        (1, 42)
    """

    n_samples, n_inputs = np.shape(angles)
    features = angles.copy()

    for q_i in range(1, n_inputs+1):
        # Find the sum of the current joint angle and all preceeding joint angles
        tot_ang = np.sum(angles[:, :q_i], axis=1)
        # Extract effects of angles on xyz based on cos & sin
        cos = np.reshape(np.cos(tot_ang), (n_samples, 1))
        sin = np.reshape(np.sin(tot_ang), (n_samples, 1))
        features = np.hstack((features, cos, sin))

    return features