from torch.utils.data import Dataset, random_split
from tqdm import tqdm
from .state import State
from .action import Action
import numpy as np
from typing import Optional
import os
import json


class TrajectoryDataset(Dataset):
    """A dataset for storing trajectories of states, actions, and rewards"""

    def __init__(self, states_continuous: np.ndarray, states_discrete: np.ndarray,
                 actions: np.ndarray, rewards: np.ndarray, terminated: np.ndarray,
                 state_paths: int = 5, state_items: int = 5, state_karts: int = 5,
                 acceleration_steps: int = 5, steer_steps: int = 7, use_reward: bool = False,
                 cfg: Optional[dict] = None):
        # Load configuration
        if cfg is None:
            config_path = os.getenv("CONFIG_PATH")
            if config_path is None:
                raise ValueError("CONFIG_PATH environment variable not set.")
            with open(config_path, 'r') as f:
                cfg = json.load(f)
        # Get discount factor
        discount_factor = cfg["discount_factor"]
        # Preprocess data
        self.states = []
        self.cumulated_rewards = np.zeros_like(rewards)
        for i in tqdm(range(len(states_continuous)), desc="Loading states"):
            # Reconstruct the State object from flattened arrays
            state = State.from_env({
                'continuous': states_continuous[i],
                'discrete': states_discrete[i]
            }, state_paths=state_paths, state_items=state_items, state_karts=state_karts)
            # Store a new flattened state
            self.states.append(State.to_model(state).unsqueeze(0))
            # Compute the cumulated reward
            if use_reward:
                cumulated_reward = 0.0
                for j in range(i, len(rewards)):
                    cumulated_reward += rewards[j] * (discount_factor ** (j - i))
                    if terminated[j]:
                        break
                self.cumulated_rewards[i] = cumulated_reward
        # Concatenate all states
        self.states = np.concatenate(self.states, axis=0)
        self.state_dim = self.states.shape[1]
        # Store actions
        self.actions = actions
        self.action_dims = Action.get_dims(acceleration_steps=acceleration_steps, steer_steps=steer_steps)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.cumulated_rewards[idx]

    def get_state_dim(self) -> int:
        """Get the dimension of the flattened state."""
        return self.state_dim

    def get_action_dims(self) -> list[int]:
        """Get dimensions of the flattened action."""
        return self.action_dims

    def train_test_split(self, test_ratio=0.2):
        """Split the dataset into training and testing sets."""
        test_size = int(len(self) * test_ratio)
        train_size = len(self) - test_size
        return random_split(self, [train_size, test_size])
