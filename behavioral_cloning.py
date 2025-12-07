import os
from core.trajectory import TrajectoryDataset
from core.nn import ActorMLP
from core.action import Action
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional
from dotenv import load_dotenv
import json
import torch
from tqdm import tqdm
import neptune
import argparse


def train(data_path: str, model_path: str, experiment_name: Optional[str] = None, nb_epochs: int = 1000, batch_size: int = 1024, learning_rate: float = 1e-3,
          test_ratio: float = 0.1, logging: bool = False, tags: list[str] = [], cfg: Optional[dict] = None):
    """Train a behavioral cloning model on the given dataset.

    Args:
        data_path (str): Path to the dataset file (npz format).
        model_path (str): Path to save the trained model.
        experiment_name (Optional[str]): Name of the experiment for logging.
        nb_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        test_ratio (float): Ratio of the dataset to use for testing.
        logging (bool): Whether to log the training process to Neptune.
        tags (list[str]): Tags for the Neptune experiment.
        cfg (Optional[dict]): Configuration dictionary. If None, it will be loaded from CONFIG_PATH environment variable.
    """
    # Load environment variables
    load_dotenv(override=True)

    # Load configuration
    if cfg is None:
        config_path = os.getenv("CONFIG_PATH")
        if config_path is None:
            raise ValueError("CONFIG_PATH environment variable not set.")
        with open(config_path, 'r') as f:
            cfg = json.load(f)

    # Load data (npz file)
    data_file = np.load(data_path)
    states = data_file['states']
    actions = data_file['actions']
    # rewards = data_file['rewards']
    # terminated = data_file['terminated']

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
            "nb_epochs": nb_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "test_ratio": test_ratio,
            **cfg
        }
        neptune_run["parameters"] = params
        # Upload a copy of the state representation source code
        neptune_run['src/state.py'].upload('core/state.py')
        # Upload a copy of the action representation source code
        neptune_run['src/action.py'].upload('core/action.py')

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create dataset
    dataset = TrajectoryDataset(states, actions)

    # Split into train and test sets
    train_dataset, test_dataset = dataset.train_test_split(test_ratio=test_ratio)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    state_dim = dataset.get_state_dim()
    model = ActorMLP(input_dim=state_dim, hidden_dim=cfg["hidden_dim"]).to(device)

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse_fn = torch.nn.MSELoss()
    bce_fn = torch.nn.BCELoss()
    action_types = Action.get_types()

    tqdm_bar = tqdm(range(nb_epochs), desc=f"Epoch 0/{nb_epochs}")
    for epoch in tqdm_bar:
        # Training step
        model.train()
        train_losses = []
        for states_batch, actions_batch in train_loader:
            states_batch = states_batch.to(device)
            actions_batch = actions_batch.to(device)

            optimizer.zero_grad()
            pred_actions = model(states_batch)

            train_loss = 0.0
            for i, (name, action_type, pred_action) in enumerate(zip(Action.NAMES, action_types, pred_actions)):
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

        # Evaluation step
        model.eval()
        test_losses = []
        with torch.no_grad():
            for states_batch, actions_batch in test_loader:
                states_batch = states_batch.to(device)
                actions_batch = actions_batch.to(device)

                pred_actions = model(states_batch)

                test_loss = 0.0
                for i, (name, action_type, pred_action) in enumerate(zip(Action.NAMES, action_types, pred_actions)):
                    # Compute loss and accuracy for each action dimension according to its type
                    if action_type == 'discrete':
                        action_loss = bce_fn(pred_action, actions_batch[:, i])
                        acc = ((pred_action >= 0.5).int() == actions_batch[:, i]).float().mean()
                        if logging:
                            neptune_run[f'test/{name}_accuracy'].append(acc.item())
                            neptune_run[f'test/{name}_loss'].append(action_loss.item())
                    else:
                        action_loss = mse_fn(pred_action, actions_batch[:, i])
                        if logging:
                            neptune_run[f'test/{name}_loss'].append(action_loss.item())
                    # Accumulate loss
                    test_loss += action_loss
                # Save and log total loss
                test_losses.append(test_loss.item())
                if logging:
                    neptune_run['test/total_loss'].append(test_loss.item())

        # Save model
        torch.save(model.state_dict(), model_path)
        # Upload model to neptune
        if logging:
            neptune_run["model/behavioral_cloning.pth"].upload(model_path)

        # Update progress bar description
        tqdm_bar.set_description(f"Epoch {epoch+1}/{nb_epochs} - Train Loss: {np.mean(train_losses).item():.2f} - Test Loss: {np.mean(test_losses).item():.2f}")

    # Stop neptune run
    if logging:
        neptune_run.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        required=True,
        help="Path to the dataset file.",
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        required=True,
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Name of the experiment for logging.",
    )
    parser.add_argument(
        "--tags",
        "-t",
        nargs="+",
        default=None,
        help="Tags for the experiment.",
    )
    parser.add_argument(
        "--nb_epochs",
        type=int,
        default=1000,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Ratio of the dataset to use for testing.",
    )
    parser.add_argument(
        "--logging",
        "-l",
        action="store_true",
        help="Enable logging to Neptune.",
    )

    args = parser.parse_args()

    train(
        data_path=args.data_path,
        model_path=args.model_path,
        experiment_name=args.name,
        tags=args.tags if args.tags is not None else [],
        nb_epochs=args.nb_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        test_ratio=args.test_ratio,
        logging=args.logging
    )
