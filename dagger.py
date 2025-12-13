"""Here's a personalized implementation of the DAgger algorithm for imitation learning.
We opt for this approach as the sb3 DAgger implementation lacks support for custom losses."""


import random
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from pystk2_gymnasium import AgentSpec
from pystk2_gymnasium.stk_wrappers import ConstantSizedObservations
from core.wrappers import EnhanceObservations, CustomPolarObservations, StuckObservations, ExpertObservations
from core.expert import ExpertAgent
from core.nn import ActorMLP, CriticMLP
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
        self.returns = []
        self.max_size = max_size

    def add(self, state: torch.Tensor, action: torch.Tensor, return_: Optional[torch.Tensor] = None):
        """Add a state-action pair to the buffer."""
        if self.size() >= self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            if return_ is not None:
                self.returns.pop(0)
        self.states.append(state)
        self.actions.append(action)
        if return_ is not None:
            self.returns.append(return_)

    def bulk_add(self, states: list[torch.Tensor], actions: list[torch.Tensor], returns: Optional[list[torch.Tensor]] = None):
        """Add multiple state-action pairs to the buffer."""
        if returns is None or len(returns) == 0:
            returns = [None] * len(states)
        for state, action, return_ in zip(states, actions, returns):
            self.add(state, action, return_)

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

    @torch.no_grad()
    def compute_gae(
        self,
        critic,
        states: torch.Tensor,          # [T + 1, obs_dim]
        rewards: torch.Tensor,         # [T]
        terminated: torch.Tensor,      # [T]  (1.0 if terminated at step t, else 0.0)
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ):
        """Compute GAE advantages and returns.

        Args:
            critic: The critic model to estimate state values.
            states (torch.Tensor): Tensor of states of shape [T + 1, obs_dim].
            rewards (torch.Tensor): Tensor of rewards of shape [T].
            terminated (torch.Tensor): Tensor indicating episode termination of shape [T].
            gamma (float): Discount factor.
            gae_lambda (float): GAE lambda parameter.
            device (str): Device to perform computations on.
        """
        # Move tensors to device
        states = states.to(device)
        rewards = rewards.to(device)
        terminated = terminated.to(device)
        # Compute all state values V(s_t)
        all_values = critic(states)        # [T + 1]
        # V(s_t)
        values = all_values[:-1]           # [T]
        # V(s_t+1)
        next_values = all_values[1:]       # [T]
        # Compute mask for bootstrapping
        must_bootstrap = 1.0 - terminated  # [T]
        # Compute TD deltas
        deltas = rewards + gamma * next_values * must_bootstrap - values
        # Compute advantages
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + gamma * gae_lambda * must_bootstrap[t] * gae
            advantages[t] = gae
        # Compute returns
        returns = advantages + values
        return advantages.detach(), returns.detach(), values.detach()

    def train(self, nb_iterations: int = 1000, epochs: int = 10, nb_races_per_iteration: int = 2,
              buffer_size: int = 100000, batch_size: int = 512, learning_rate: float = 1e-3, min_eps: float = 1e-3,
              with_critic: bool = False, gamma: float = 0.99, gae_lambda: float = 0.95, policy_model_path: str = "dagger_policy.pth",
              critic_model_path: str = "dagger_critic.pth", logging: bool = False, experiment_name: Optional[str] = None,
              tags: list[str] = []):
        """Train the model using the DAgger algorithm.

        Args:
            nb_iterations (int): Number of DAgger iterations.
            epochs (int): Number of training epochs per iteration.
            nb_races_per_iteration (int): Number of races to run per iteration.
            buffer_size (int): Maximum size of the replay buffer.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the optimizer.
            min_eps (float): Minimum epsilon for expert action selection.
            with_critic (bool): Whether to use a critic for advantage estimation.
            gamma (float): Discount factor for GAE.
            gae_lambda (float): GAE lambda parameter.
            policy_model_path (str): Path to save the trained policy model.
            critic_model_path (str): Path to save the trained critic model.
            logging (bool): Whether to log the training process to Neptune.
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
            lambda env: TimeLimit(env, max_episode_steps=2500),
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
        policy_model = ActorMLP(input_dim=State.to_model(State.from_env(state)).shape[0], hidden_dim=512).to(device)
        if with_critic:
            critic_model = CriticMLP(input_dim=State.to_model(State.from_env(state)).shape[0], hidden_dim=512).to(device)
        # Prepare the replay buffer
        replay_buffer = ReplayBuffer(max_size=buffer_size)
        # Prepare the optimizer and loss function
        policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
        if with_critic:
            critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=learning_rate)
        mse_fn = torch.nn.MSELoss()
        bce_fn = torch.nn.BCELoss()
        action_types = Action.get_types()
        # Start DAgger iterations
        eps = 1.0
        eps_rate = min_eps ** (1 / nb_iterations)
        tqdm_bar = tqdm(range(nb_iterations), desc="DAgger Training")
        for it in tqdm_bar:
            # Set model to eval mode
            policy_model.eval()
            # Run several races to collect data
            for _ in range(nb_races_per_iteration):
                # Initialize episode data
                episode_states = []
                episode_actions = []
                episode_rewards = []
                episode_terminated = []
                # Rollout episode
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
                            action_pred = torch.cat(policy_model(model_input.to(device))).cpu()
                            action_to_play = Action.to_env(Action.from_model(action_pred))
                    # Step in the environment
                    state, reward, terminated, truncated, _ = env.step(action_to_play)
                    done = terminated or truncated
                    # Store episode data
                    episode_states.append(state_tensor)
                    episode_actions.append(expert_action_tensor)
                    episode_rewards.append(reward)
                    episode_terminated.append(float(terminated))
                # If we are training a critic, compute episode return (GAE)
                if with_critic:
                    episode_states.append(State.to_model(State.from_env(state)))  # Add final state
                    states_tensor = torch.stack(episode_states)                                # [T + 1, obs_dim]
                    rewards_tensor = torch.tensor(episode_rewards, dtype=torch.float32)        # [T]
                    terminated_tensor = torch.tensor(episode_terminated, dtype=torch.float32)  # [T]
                    advantages, returns, _ = self.compute_gae(
                        critic_model,
                        states_tensor,
                        rewards_tensor,
                        terminated_tensor,
                        gamma=gamma,
                        gae_lambda=gae_lambda,
                        device=device
                    )
                    # Store the state-expert_action-return triplet in the replay buffer
                    replay_buffer.bulk_add(episode_states[:-1], episode_actions, returns)
                else:
                    # Store the state-expert_action pair in the replay buffer
                    replay_buffer.bulk_add(episode_states, episode_actions)
            # Decay epsilon
            eps *= eps_rate
            # Prepare the dataset from the replay buffer
            dataset = TrajectoryDataset(replay_buffer.states, replay_buffer.actions, replay_buffer.returns)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            # Set model to train mode
            policy_model.train()
            train_stats = {
                'policy_loss': []
            }
            if with_critic:
                critic_model.train()
                train_stats['critic_loss'] = []
            # Train the model for several epochs
            for epoch in range(epochs):
                # Training step
                for states_batch, actions_batch, returns_batch in train_loader:
                    # Move batch to device
                    states_batch = states_batch.to(device)
                    actions_batch = actions_batch.to(device)
                    returns_batch = returns_batch.to(device)
                    # Policy forward pass
                    policy_optimizer.zero_grad()
                    pred_actions = policy_model(states_batch)
                    # Critic forward pass
                    if with_critic:
                        critic_optimizer.zero_grad()
                        values = critic_model(states_batch)
                    # Compute policy loss
                    policy_loss = 0.0
                    for i, ((name, action_type), pred_action) in enumerate(zip(action_types, pred_actions)):
                        # Compute loss and accuracy for each action dimension according to its type
                        if action_type == 'discrete':
                            action_loss = bce_fn(pred_action, actions_batch[:, i])
                            acc = ((pred_action >= 0.5).int() == actions_batch[:, i]).float().mean()
                            if logging:
                                if f'{name}_accuracy' not in train_stats:
                                    train_stats[f'{name}_accuracy'] = []
                                train_stats[f'{name}_accuracy'].append(acc.item())
                                if f'{name}_loss' not in train_stats:
                                    train_stats[f'{name}_loss'] = []
                                train_stats[f'{name}_loss'].append(action_loss.item())
                        else:
                            action_loss = mse_fn(pred_action, actions_batch[:, i])
                            if logging:
                                if f'{name}_loss' not in train_stats:
                                    train_stats[f'{name}_loss'] = []
                                train_stats[f'{name}_loss'].append(action_loss.item())
                        # Accumulate loss
                        policy_loss += action_loss
                    # Save policy loss
                    train_stats['policy_loss'].append(policy_loss.item())
                    total_loss = policy_loss
                    # Compute critic loss
                    if with_critic:
                        value_loss = mse_fn(values, returns_batch)
                        train_stats['critic_loss'].append(value_loss.item())
                        total_loss += value_loss
                    # Backpropagation
                    total_loss.backward()
                    policy_optimizer.step()
                    if with_critic:
                        critic_optimizer.step()
            # Log training metrics to neptune
            if logging:
                for key, values in train_stats.items():
                    neptune_run[f"train/{key}"].log(np.mean(values).item())
            # Save model
            if it % 10 == 0:
                # Save policy model locally
                torch.save(policy_model.state_dict(), policy_model_path)
                # Save critic model locally
                if with_critic:
                    torch.save(critic_model.state_dict(), critic_model_path)
                # Upload models to neptune
                if logging:
                    # Upload policy model to neptune
                    neptune_run["model/dagger_policy"].upload(policy_model_path)
                    # Upload critic model to neptune
                    if with_critic:
                        neptune_run["model/dagger_critic"].upload(critic_model_path)
            # Update progress bar description
            tqdm_bar.set_description(f"Iteration {it+1}/{nb_iterations} - Train Loss: {np.mean(train_stats['policy_loss']).item():.2f}")

        # Save final policy model locally
        torch.save(policy_model.state_dict(), policy_model_path)
        # Save final critic model locally
        if with_critic:
            torch.save(critic_model.state_dict(), critic_model_path)
        # Upload final models to neptune and stop neptune run
        if logging:
            # Upload final policy model to neptune
            neptune_run["model/dagger_policy"].upload(policy_model_path)
            # Upload final critic model to neptune
            if with_critic:
                neptune_run["model/dagger_critic"].upload(critic_model_path)
            # Stop neptune run
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
    parser.add_argument("--with_critic", action="store_true", help="Whether to use a critic for advantage estimation.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for GAE.")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter.")
    parser.add_argument("--policy_model_path", type=str, default="dagger_policy.pth", help="Path to save the trained policy model.")
    parser.add_argument("--critic_model_path", type=str, default="dagger_critic.pth", help="Path to save the trained critic model.")
    parser.add_argument("--logging", '-l', action="store_true", help="Whether to log the training process to Neptune.")
    parser.add_argument("--experiment_name", type=str, default=None, help="Name of the experiment for logging.")
    parser.add_argument("--tags", nargs='*', default=[], help="Tags for the Neptune experiment.")
    parser.add_argument("--disable_drift", '-d', action="store_true", help="Whether to enable drifting behavior in the expert agent.")
    args = parser.parse_args()

    dagger_trainer = DAgger(enable_drift=not args.disable_drift)
    dagger_trainer.train(
        nb_iterations=args.nb_iterations,
        epochs=args.epochs,
        nb_races_per_iteration=args.nb_races_per_iteration,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_eps=args.min_eps,
        with_critic=args.with_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        policy_model_path=args.policy_model_path,
        critic_model_path=args.critic_model_path,
        logging=args.logging,
        experiment_name=args.experiment_name,
        tags=args.tags
    )
