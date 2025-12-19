import torch
from .nn import ActorMLP
from .state import State
from .action import Action
from typing import Optional


class Actor:
    """Actor class that uses a pre-trained neural network to predict actions based on input states."""

    model: ActorMLP
    """The neural network model used for action prediction."""

    def __init__(self, model_path: str, input_dim: int, hidden_dim: int, head_dim: Optional[int] = None):
        """Load a pre-trained actor model from the specified path.

        Args:
            model_path (str): Path to the saved model weights.
            input_dim (int): Dimension of the input state.
            hidden_dim (int): Dimension of the hidden layers in the model.
        """
        self.model = ActorMLP(input_dim, hidden_dim, head_dim=head_dim)
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
        self.model.eval()

    def act(self, input_state: dict, all_features: bool = False):
        """Given an input state from the environment, predict the action to take.

        Args:
            input_state (dict): The state representation from the environment.

        Returns:
            dict: The action representation suitable for the environment.
        """
        state = State.from_env(input_state)
        pred_action = torch.cat(self.model(State.to_model(state, all_features=all_features).unsqueeze(0)))
        action = Action.from_model(pred_action)
        return Action.to_env(action)
