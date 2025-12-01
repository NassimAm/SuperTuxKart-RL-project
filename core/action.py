from typing import TypedDict
import numpy as np
import torch


class Action(TypedDict):
    """Base class for actions in the environment."""

    acceleration: float = 0.0
    """Acceleration component of the action between 0.0 and 1.0."""
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
    steer: float = 0.0
    """Steering component of the action between -1.0 and 1.0 (positive -> right, negative -> left)."""
    look_back: bool = False  # TODO: see how to integrate this in the action space
    """Look back component of the action."""

    NAMES = ['acceleration', 'brake', 'drift', 'fire', 'nitro', 'rescue', 'steer']
    CONTINUOUS_ACTIONS = ['acceleration', 'steer']
    DISCRETE_ACTIOSNS = ['brake', 'drift', 'fire', 'nitro', 'rescue']

    def from_env(input_action: dict) -> 'Action':
        """Create an Action from a list representation."""
        return Action(
            acceleration=input_action['acceleration'].item(),
            brake=bool(input_action['brake']),
            drift=bool(input_action['drift']),
            fire=bool(input_action['fire']),
            nitro=bool(input_action['nitro']),
            rescue=bool(input_action['rescue']),
            steer=input_action['steer'].item(),
            look_back=False
        )

    def empty() -> 'Action':
        """Create an empty Action with all components set to their default values."""
        return Action(
            acceleration=0.0,
            brake=False,
            drift=False,
            fire=False,
            nitro=False,
            rescue=False,
            steer=0.0,
            look_back=False
        )

    def to_env(action: 'Action') -> np.ndarray:
        """Convert the action to the environment's expected format."""
        return {
            'acceleration': np.array(action['acceleration']),
            'brake': int(action['brake']),
            'drift': int(action['drift']),
            'fire': int(action['fire']),
            'nitro': int(action['nitro']),
            'rescue': int(action['rescue']),
            'steer': np.array(action['steer'])
        }

    def from_model(model_output: torch.Tensor) -> 'Action':
        """Create an Action from the model's output tensor."""
        return Action(
            acceleration=model_output[0].item(),
            brake=(model_output[1].item() >= 0.5),
            drift=(model_output[2].item() >= 0.5),
            fire=(model_output[3].item() >= 0.5),
            nitro=(model_output[4].item() >= 0.5),
            rescue=(model_output[5].item() >= 0.5),
            steer=model_output[6].item(),
            look_back=False
        )

    def to_model(action: 'Action') -> torch.Tensor:
        """Convert the action to the model's expected format."""
        return torch.cat([
            torch.tensor([action['acceleration']], dtype=torch.float32),
            torch.tensor([float(action['brake'])], dtype=torch.float32),
            torch.tensor([float(action['drift'])], dtype=torch.float32),
            torch.tensor([float(action['fire'])], dtype=torch.float32),
            torch.tensor([float(action['nitro'])], dtype=torch.float32),
            torch.tensor([float(action['rescue'])], dtype=torch.float32),
            torch.tensor([action['steer']], dtype=torch.float32),
        ])

    def get_types() -> list[str]:
        """Get the dimensions of each action component."""
        return [
            'continuous-sigmoid',  # acceleration
            'discrete',  # brake
            'discrete',  # drift
            'discrete',  # fire
            'discrete',  # nitro
            'discrete',  # rescue
            'continuous-tanh',  # steer
        ]

    def acceleration_to_continuous(acceleration_discrete: int, acceleration_steps: int = 5) -> float:
        """Convert discrete acceleration to continuous value between 0 and 1."""
        assert acceleration_steps > 1, "acceleration_steps must be greater than 1"
        if isinstance(acceleration_discrete, np.ndarray):
            acceleration_discrete = acceleration_discrete.item()
        return (acceleration_discrete / (acceleration_steps - 1))

    def acceleration_to_discrete(acceleration_continuous: float, acceleration_steps: int = 5) -> int:
        """Convert continuous acceleration value between 0 and 1 to discrete."""
        assert acceleration_steps > 1, "acceleration_steps must be greater than 1"
        if isinstance(acceleration_continuous, np.ndarray):
            acceleration_continuous = acceleration_continuous.item()
        clamped_value = max(0.0, min(1.0, acceleration_continuous))
        return int(round(clamped_value * (acceleration_steps - 1)))

    def steer_to_continuous(steer_discrete: int, steer_steps: int = 7) -> float:
        """Convert discrete steer to continuous value between -1 and 1."""
        assert (steer_steps > 1) and (steer_steps % 2 == 1), "steer_steps must be an odd number greater than 1"
        if isinstance(steer_discrete, np.ndarray):
            steer_discrete = steer_discrete.item()
        return (steer_discrete / (steer_steps - 1)) * 2.0 - 1.0

    def steer_to_discrete(steer_continuous: float, steer_steps: int = 7) -> int:
        """Convert continuous steer value between -1 and 1 to discrete."""
        assert (steer_steps > 1) and (steer_steps % 2 == 1), "steer_steps must be an odd number greater than 1"
        if isinstance(steer_continuous, np.ndarray):
            steer_continuous = steer_continuous.item()
        clamped_value = max(-1.0, min(1.0, steer_continuous))
        return int(round(((clamped_value + 1.0) / 2.0) * (steer_steps - 1)))
