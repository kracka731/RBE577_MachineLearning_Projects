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
        print(f"phys objs {physics_config.keys()}")
        mass, size, inertia_factor, _, _ = physics_config.keys()

        # Set simulation parameters
        instance = PushPhysics(mass, size, inertia_factor)
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
        rotation = push_params.numpy()[:, 0]
        side     = push_params.numpy()[:, 1]
        distance = push_params.numpy()[:, 2]

        # Compute velocity profile
        v_max = 2*distance / duration
        v = lambda t: v_max * (0.5*np.sin(((2*np.pi*t)/duration) - (np.pi/2)) + 0.5)

        # Initialize states (x, y, theta)
        x_local, y_local, theta = 0

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
            delta_x = -v_i * np.cos(theta) * delta_t
            delta_y = -v_i * np.sin(theta) * delta_t
            x_local += delta_x
            y_local += delta_y

        # Transform local frame motion to global frame
        R = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
        xy = R * np.array([[x_local], [y_local]])
        x_global = xy[0, 0]
        y_global = xy[1, 0]

        return torch.stack([x_global, y_global, theta], dim=1)
