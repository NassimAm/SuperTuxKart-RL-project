from torch.utils.data import Dataset, random_split
import numpy as np
import torch


class TrajectoryDataset(Dataset):
    """A dataset for storing trajectories of states, actions, and rewards"""

    def __init__(self, states: np.ndarray, actions: np.ndarray):
        self.states = torch.from_numpy(states)
        self.actions = torch.from_numpy(actions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

    def get_state_dim(self) -> int:
        """Get the dimension of the flattened state."""
        return self.states.shape[1]

    def get_action_dim(self) -> int:
        """Get dimensions of the flattened action."""
        return self.actions.shape[1]

    def train_test_split(self, test_ratio=0.2):
        """Split the dataset into training and testing sets."""
        test_size = int(len(self) * test_ratio)
        train_size = len(self) - test_size
        return random_split(self, [train_size, test_size])
