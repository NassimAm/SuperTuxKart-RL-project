import gymnasium as gym
from bbrl.agents import Agent
import torch
from .core.state import State
from .core.action import Action
from .core.nn import ActorMLP
# from .core.expert import ExpertAgent


class MyWrapper(gym.ActionWrapper):
    def __init__(self, env, option: int):
        super().__init__(env)
        self.option = option

    def action(self, action):
        # We do nothing here
        return action


class Actor(Agent):
    """Computes probabilities over action"""

    def __init__(self, observation_space, action_space):
        super().__init__()
        # self.expert = ExpertAgent()
        self.model = ActorMLP(input_dim=179, hidden_dim=512)

    def load_state_dict(self, state: dict):
        self.model.load_state_dict(state)

    def forward(self, t: int):
        obs = {key: self.get((f'env/env_obs/{key}', t))[0].numpy() for key in State.KEYS}
        state_obj = State.from_env(obs)
        state_tensor = State.to_model(state_obj).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            pred_action = torch.cat(self.model(state_tensor))
            action = Action.to_env(Action.from_model(pred_action))
        # action = self.expert.act(obs)
        for key, value in action.items():
            self.set((f'action/{key}', t), torch.tensor([value]))


class ArgmaxActor(Agent):
    """Actor that computes the action"""

    def forward(self, t: int):
        # Selects the best actions according to the policy
        pass


class SamplingActor(Agent):
    """Just sample random actions"""

    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space

    def forward(self, t: int):
        action = self.action_space.sample()
        for key, value in action.items():
            self.set((f'action/{key}', t), torch.tensor([value.item()]))
