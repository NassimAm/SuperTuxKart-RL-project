from torch.utils.data import Dataset, random_split
import numpy as np
import torch


class TrajectoryDataset(Dataset):
    """A dataset for storing trajectories of states, actions, and rewards"""

    def __init__(self, states, actions, returns=None):
        if isinstance(states, np.ndarray) and isinstance(actions, np.ndarray):
            self.states = torch.from_numpy(states)
            self.actions = torch.from_numpy(actions)
            if (returns is not None) and (len(returns) > 0) and isinstance(returns, np.ndarray):
                self.returns = torch.from_numpy(returns)
            else:
                self.returns = torch.zeros((self.states.shape[0],))
        else:
            self.states = torch.cat([s.unsqueeze(0) for s in states], dim=0)
            self.actions = torch.cat([a.unsqueeze(0) for a in actions], dim=0)
            if (returns is not None) and (len(returns) > 0):
                self.returns = torch.cat([r.unsqueeze(0) for r in returns], dim=0)
            else:
                self.returns = torch.zeros((self.states.shape[0],))

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.returns[idx]

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
