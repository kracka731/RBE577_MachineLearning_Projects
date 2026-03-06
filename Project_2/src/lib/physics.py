import torch
import numpy as np
from typing import Dict, Any


class PushPhysics:
    """Physics engine for push interactions"""

    def __init__(
        self, mass: float = 0.1, size: float = 0.1, inertia_factor: float = 1 / 12
    ):
        # Object properties
        self.mass = float(mass)
        self.size = float(size)
        self.inertia_factor = float(inertia_factor)
        self.inertia = self.inertia_factor * self.mass * (self.size**2)

        # Default simulation parameters
        self._push_duration = 3.0
        self._simulation_steps = 100

    @classmethod
    def from_config(cls, physics_config: Dict[str, Any]) -> "PushPhysics":
        """Create PushPhysics instance from config dictionary"""
        # Extract object properties from config
        mass, size, inertia_factor, dur, steps = physics_config.values()

        # Set simulation parameters
        instance = PushPhysics(mass, size, inertia_factor)
        instance._push_duration = dur
        instance._simulation_steps = steps
        return instance

    def compute_motion(
        self, push_params: torch.Tensor, duration: float = None, steps: int = None
    ) -> torch.Tensor:
        """
        Compute object motion given push parameters

        Args:
            push_params: [batch_size, 3] tensor of [rotation, side, distance]
            duration: Duration of push in seconds (optional)
            steps: Number of simulation steps (optional)

        Returns:
            [batch_size, 3] tensor of [x, y, theta] final states
        """
        # Define motion duration and steps
        if duration is None: 
            duration = self._push_duration
        if steps is None: 
            steps = self._simulation_steps

        # Extract push parameters (rotation, side, distance)
        batch_size, _ = tuple(push_params.shape)
        rotation = push_params[:, 0]
        side     = push_params[:, 1]
        distance = push_params[:, 2]

        # Compute velocity profile
        v_max = 2*distance / duration
        v = lambda t: v_max * (0.5*np.sin(((2*np.pi*t)/duration) - (np.pi/2)) + 0.5)

        # Initialize states (x, y, theta)
        x_local = y_local = theta = torch.zeros(rotation.shape)

        # Loop through simulation steps to update states
        delta_t = duration / steps
        for t in range(2, steps+1):
            v_i = v(t*delta_t)
            torque = self.mass * v_i * side 

            # Angular update
            alpha = torque / self.inertia # angular acceleration 
            delta_theta = 0.5 * alpha * delta_t**2 
            theta += delta_theta

            # Position update 
            delta_x = v_i * torch.cos(theta) * delta_t
            delta_y = v_i * torch.sin(theta) * delta_t
            x_local += delta_x
            y_local += delta_y

        xy_local = torch.stack([x_local, y_local], dim=1).reshape(batch_size, 2, 1) # shape = (n,2,1)

        # Evaluate rotation between local/global frames
        cos_r = torch.cos(rotation).reshape(batch_size, 1, 1)
        sin_r = torch.sin(rotation).reshape(batch_size, 1, 1)
        
        R_top_row = torch.cat([cos_r, -sin_r], dim=2) # shape = (n,1,2)
        R_bot_row = torch.cat([sin_r, cos_r], dim=2)  # shape = (n,1,2)
        R = torch.cat([R_top_row, R_bot_row], dim=1)  # shape = (n,2,2)

        # Transform local frame motion to global frame
        xy_global = R @ xy_local       # shape = (n,2,1) = (n,2,2) * (n,2,1)
        x_global = xy_global[:, 0, 0]  # shape = (n)
        y_global = xy_global[:, 1, 0]  # shape = (n)

        return torch.stack([x_global, y_global, theta], dim=1)

    def physics_pred(self, loaded_dataset):
        """Physics-based predictions for entire dataset"""
        pred_list = []
        mse_list = []
        for batch, (X, y) in enumerate(loaded_dataset):
            # print(f"batch: {batch}")
            predictions = self.compute_motion(X)
            # print(f"predictions: {predictions}")
            mse = torch.mean((predictions - y) ** 2)
            # print(f"MSE for one batch: {mse}")

            pred_list.extend(predictions)
            mse_list.extend([mse.item()])

        # print(len(pred_list))
        avg_mse = sum(mse_list) / len(mse_list)
        print(f"avg_mse: {avg_mse}")

        return pred_list