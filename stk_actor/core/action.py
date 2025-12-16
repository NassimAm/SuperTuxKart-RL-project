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

    def from_env(input_action: dict) -> 'Action':
        """Create an Action from a list representation."""
        return Action(
            acceleration=input_action['acceleration'].item() if isinstance(input_action['acceleration'], np.ndarray) else input_action['acceleration'],
            brake=bool(input_action['brake'].item()) if isinstance(input_action['brake'], np.ndarray) else bool(input_action['brake']),
            drift=bool(input_action['drift'].item()) if isinstance(input_action['drift'], np.ndarray) else bool(input_action['drift']),
            fire=bool(input_action['fire'].item()) if isinstance(input_action['fire'], np.ndarray) else bool(input_action['fire']),
            nitro=bool(input_action['nitro'].item()) if isinstance(input_action['nitro'], np.ndarray) else bool(input_action['nitro']),
            rescue=bool(input_action['rescue'].item()) if isinstance(input_action['rescue'], np.ndarray) else bool(input_action['rescue']),
            steer=input_action['steer'].item() if isinstance(input_action['steer'], np.ndarray) else input_action['steer'],
            look_back=False
        )

    def to_env(action: 'Action'):
        """Convert the action to the environment's expected format."""
        return {
            'acceleration': action['acceleration'],
            'brake': int(action['brake']),
            'drift': int(action['drift']),
            'fire': int(action['fire']),
            'nitro': int(action['nitro']),
            'rescue': int(action['rescue']),
            'steer': action['steer']
        }

    def from_array(input_array: np.ndarray) -> 'Action':
        """Create an Action from a list representation."""
        return Action(
            acceleration=input_array[0].item(),
            brake=bool(input_array[1].item()),
            drift=bool(input_array[2].item()),
            fire=bool(input_array[3].item()),
            nitro=bool(input_array[4].item()),
            rescue=bool(input_array[5].item()),
            steer=input_array[6].item(),
            look_back=False
        )

    def to_array(action: 'Action') -> np.ndarray:
        """Convert the action to a list representation."""
        return np.array([
            action['acceleration'],
            int(action['brake']),
            int(action['drift']),
            int(action['fire']),
            int(action['nitro']),
            int(action['rescue']),
            action['steer']
        ], dtype=np.float32)

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

    def get_types() -> list[tuple[str, str]]:
        """Get the dimensions of each action component."""
        return [
            ('acceleration', 'continuous-sigmoid'),  # acceleration
            ('brake', 'discrete'),  # brake
            ('drift', 'discrete'),  # drift
            ('fire', 'discrete'),  # fire
            ('nitro', 'discrete'),  # nitro
            ('rescue', 'discrete'),  # rescue
            ('steer', 'continuous-tanh'),  # steer
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
