"""Here's a personalized implementation of the DAgger algorithm for imitation learning.
We opt for this approach as the sb3 DAgger implementation lacks support for custom losses."""


import random
import gymnasium as gym
from pystk2_gymnasium import AgentSpec
from pystk2_gymnasium.stk_wrappers import ConstantSizedObservations
from core.wrappers import EnhanceObservations, CustomPolarObservations, StuckObservations, ExpertObservations
from core.expert import ExpertAgent
from core.nn import ActorMLP
from core.state import State
from core.action import Action
from core.trajectory import TrajectoryDataset
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from dotenv import load_dotenv
import neptune
from typing import Optional
import argparse


class ReplayBuffer:
    """A simple replay buffer to store state-action pairs."""

    def __init__(self, max_size: int = 100000):
        self.states = []
        self.actions = []
        self.max_size = max_size

    def add(self, state, action):
        """Add a state-action pair to the buffer."""
        if self.size() >= self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
        self.states.append(state)
        self.actions.append(action)

    def size(self):
        """Return the current size of the buffer."""
        return len(self.states)


class DAgger:
    """DAgger algorithm for imitation learning."""

    def __init__(self, enable_drift: bool = True):
        """Initialize the DAgger algorithm components.

        Args:
            enable_drift (bool): Whether to enable drifting behavior in the expert agent.
        """
        self.enable_drift = enable_drift

    def train(self, nb_iterations: int = 1000, epochs: int = 10, nb_races_per_iteration: int = 2,
              buffer_size: int = 100000, batch_size: int = 512, learning_rate: float = 1e-3, min_eps: float = 1e-3,
              logging: bool = False, model_path: str = "dagger.pth", experiment_name: Optional[str] = None, tags: list[str] = []):
        """Train the model using the DAgger algorithm.

        Args:
            nb_iterations (int): Number of DAgger iterations.
            epochs (int): Number of training epochs per iteration.
            nb_races_per_iteration (int): Number of races to run per iteration.
            buffer_size (int): Maximum size of the replay buffer.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the optimizer.
            min_eps (float): Minimum epsilon for expert action selection.
            logging (bool): Whether to log the training process to Neptune.
            model_path (str): Path to save the trained model.
            experiment_name (Optional[str]): Name of the experiment for logging.
            tags (list[str]): Tags for the Neptune experiment.
        """
        # Load environment variables
        load_dotenv(override=True)
        # Setup neptune for logging if enabled
        if logging:
            neptune_project = os.getenv("NEPTUNE_PROJECT")
            neptune_api_token = os.getenv("NEPTUNE_API_TOKEN")
            if neptune_project is None:
                raise ValueError("NEPTUNE_PROJECT environment variable not set.")
            if neptune_api_token is None:
                raise ValueError("NEPTUNE_API_TOKEN environment variable not set.")
            neptune_run = neptune.init_run(
                project=neptune_project,
                api_token=neptune_api_token,
                name=experiment_name,
                tags=tags
            )
            params = {
                "nb_epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "hidden_dim": 512
            }
            neptune_run["parameters"] = params
            # Upload a copy of the state representation source code
            neptune_run['src/state.py'].upload('core/state.py')
            # Upload a copy of the action representation source code
            neptune_run['src/action.py'].upload('core/action.py')
        # Initialize the environment with necessary wrappers
        env = gym.make("supertuxkart/full-v0", render_mode=None, agent=AgentSpec(use_ai=False), difficulty=2, num_kart=10)
        wrappers = [
            lambda env: ConstantSizedObservations(env, state_paths=10),
            lambda env: EnhanceObservations(env),
            lambda env: StuckObservations(env),
            lambda env: ExpertObservations(env),
            lambda env: CustomPolarObservations(env)
        ]
        for wrapper in wrappers:
            env = wrapper(env)
        # Initialize the expert agent
        expert_agent = ExpertAgent(env, enable_drift=self.enable_drift)
        # Check device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        # Initialize the actor model
        state, _ = env.reset()
        model = ActorMLP(input_dim=State.to_model(State.from_env(state)).shape[0], hidden_dim=512).to(device)
        # Prepare the replay buffer
        replay_buffer = ReplayBuffer(max_size=buffer_size)
        # Prepare the optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        mse_fn = torch.nn.MSELoss()
        bce_fn = torch.nn.BCELoss()
        action_types = Action.get_types()
        # Start DAgger iterations
        eps = 1.0
        eps_rate = min_eps ** (1 / nb_iterations)
        tqdm_bar = tqdm(range(nb_iterations), desc="DAgger Training")
        for it in tqdm_bar:
            # Set model to eval mode
            model.eval()
            # Run several races to collect data
            for _ in range(nb_races_per_iteration):
                done = False
                state, _ = env.reset()
                while not done:
                    # Get the expert action
                    expert_action = expert_agent.act(state)
                    # Convert state to tensor
                    state_tensor = State.to_model(State.from_env(state))
                    # Convert expert action to tensor
                    expert_action_tensor = Action.to_model(Action.from_env(expert_action))
                    # Get the expert action with probability eps
                    if random.random() < eps:
                        action_to_play = expert_action
                    else:
                        with torch.no_grad():
                            # Get the model input
                            model_input = state_tensor.unsqueeze(0)
                            # Get the action from the current policy
                            action_pred = torch.cat(model(model_input))
                            action_to_play = Action.to_env(Action.from_model(action_pred))
                    # Store the state-expert_action pair in the replay buffer
                    replay_buffer.add(state_tensor, expert_action_tensor)
                    # Step in the environment
                    state, reward, terminated, truncated, _ = env.step(action_to_play)
                    done = terminated or truncated
            # Decay epsilon
            eps *= eps_rate
            # Prepare the dataset from the replay buffer
            dataset = TrajectoryDataset(replay_buffer.states, replay_buffer.actions)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            # Set model to train mode
            model.train()
            train_losses = []
            # Train the model for several epochs
            for epoch in range(epochs):
                # Training step
                for states_batch, actions_batch in train_loader:
                    states_batch = states_batch.to(device)
                    actions_batch = actions_batch.to(device)

                    optimizer.zero_grad()
                    pred_actions = model(states_batch)

                    train_loss = 0.0
                    for i, (name, (_, action_type), pred_action) in enumerate(zip(Action.NAMES, action_types, pred_actions)):
                        # Compute loss and accuracy for each action dimension according to its type
                        if action_type == 'discrete':
                            action_loss = bce_fn(pred_action, actions_batch[:, i])
                            acc = ((pred_action >= 0.5).int() == actions_batch[:, i]).float().mean()
                            if logging:
                                neptune_run[f'train/{name}_accuracy'].append(acc.item())
                                neptune_run[f'train/{name}_loss'].append(action_loss.item())
                        else:
                            action_loss = mse_fn(pred_action, actions_batch[:, i])
                            if logging:
                                neptune_run[f'train/{name}_loss'].append(action_loss.item())
                        # Accumulate loss
                        train_loss += action_loss
                    # Save and log total loss
                    train_losses.append(train_loss.item())
                    if logging:
                        neptune_run['train/total_loss'].append(train_loss.item())
                    # Backpropagation
                    train_loss.backward()
                    optimizer.step()
            # Save model
            torch.save(model.state_dict(), model_path)
            # Upload model to neptune
            if logging:
                neptune_run["model/dagger"].upload(model_path)

            # Update progress bar description
            tqdm_bar.set_description(f"Iteration {it+1}/{nb_iterations} - Train Loss: {np.mean(train_losses).item():.2f}")
        # Stop neptune run
        if logging:
            neptune_run.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DAgger agent.")
    parser.add_argument("--nb_iterations", type=int, default=1000, help="Number of DAgger iterations.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs per iteration.")
    parser.add_argument("--nb_races_per_iteration", type=int, default=2, help="Number of races to run per iteration.")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Maximum size of the replay buffer.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--min_eps", type=float, default=1e-3, help="Minimum epsilon for expert action selection.")
    parser.add_argument("--logging", '-l', action="store_true", help="Whether to log the training process to Neptune.")
    parser.add_argument("--model_path", type=str, default="dagger.pth", help="Path to save the trained model.")
    parser.add_argument("--experiment_name", type=str, default=None, help="Name of the experiment for logging.")
    parser.add_argument("--tags", nargs='*', default=[], help="Tags for the Neptune experiment.")
    args = parser.parse_args()

    dagger_trainer = DAgger(enable_drift=True)
    dagger_trainer.train(
        nb_iterations=args.nb_iterations,
        epochs=args.epochs,
        nb_races_per_iteration=args.nb_races_per_iteration,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_eps=args.min_eps,
        logging=args.logging,
        model_path=args.model_path,
        experiment_name=args.experiment_name,
        tags=args.tags
    )
