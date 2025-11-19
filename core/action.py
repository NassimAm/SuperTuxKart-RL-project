from typing import TypedDict
import numpy as np
import torch


class Action(TypedDict):
    """Base class for actions in the environment."""

    acceleration: int = 0
    """Acceleration component of the action."""
    brake: bool = False
    """Brake component of the action."""
    drift: bool = False
    """Drift component of the action."""
    fire: bool = False
    """Fire component of the action."""
    nitro: bool = False
    """Nitro boost component of the action."""
    rescue: bool = False
    """Rescue component of the action."""
    steer: int = 0
    """Steering component of the action."""

    BRAKE_DIM = 2
    DRIFT_DIM = 2
    FIRE_DIM = 2
    NITRO_DIM = 2
    RESCUE_DIM = 2

    NAMES = [
        'acceleration',
        'brake',
        'drift',
        'fire',
        'nitro',
        'rescue',
        'steer'
    ]

    def from_env(array: np.ndarray) -> 'Action':
        """Create an Action from a list representation."""
        return Action(
            acceleration=int(array[0]),
            brake=bool(array[1]),
            drift=bool(array[2]),
            fire=bool(array[3]),
            nitro=bool(array[4]),
            rescue=bool(array[5]),
            steer=int(array[6]),
        )

    def to_env(action: 'Action') -> np.ndarray:
        """Convert the action to the environment's expected format."""
        return np.array([
            action['acceleration'],
            int(action['brake']),
            int(action['drift']),
            int(action['fire']),
            int(action['nitro']),
            int(action['rescue']),
            action['steer'],
        ])

    def to_model(action: 'Action', acceleration_steps: int = 5, steer_steps: int = 5) -> list[torch.Tensor]:
        """Convert the action to the model's expected format."""
        # Acceleration one-hot encoding
        acceleration_one_hot = torch.zeros(acceleration_steps)
        acceleration_one_hot[action['acceleration']] = 1.0
        # Brake one-hot encoding
        brake_one_hot = torch.zeros(Action.BRAKE_DIM)
        brake_one_hot[int(action['brake'])] = 1.0
        # Drift one-hot encoding
        drift_one_hot = torch.zeros(Action.DRIFT_DIM)
        drift_one_hot[int(action['drift'])] = 1.0
        # Fire one-hot encoding
        fire_one_hot = torch.zeros(Action.FIRE_DIM)
        fire_one_hot[int(action['fire'])] = 1.0
        # Nitro one-hot encoding
        nitro_one_hot = torch.zeros(Action.NITRO_DIM)
        nitro_one_hot[int(action['nitro'])] = 1.0
        # Rescue one-hot encoding
        rescue_one_hot = torch.zeros(Action.RESCUE_DIM)
        rescue_one_hot[int(action['rescue'])] = 1.0
        # Steering one-hot encoding
        steer_one_hot = torch.zeros(steer_steps)
        steer_one_hot[action['steer']] = 1.0
        # Return as list of arrays
        return [
            acceleration_one_hot,
            brake_one_hot,
            drift_one_hot,
            fire_one_hot,
            nitro_one_hot,
            rescue_one_hot,
            steer_one_hot,
        ]

    def get_dims(acceleration_steps: int = 5, steer_steps: int = 7) -> list[int]:
        """Get the dimensions of each action component."""
        return [
            acceleration_steps,
            Action.BRAKE_DIM,
            Action.DRIFT_DIM,
            Action.FIRE_DIM,
            Action.NITRO_DIM,
            Action.RESCUE_DIM,
            steer_steps,
        ]
