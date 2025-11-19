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
          test_ratio: float = 0.1, logging: bool = False, cfg: Optional[dict] = None):
    """Train a behavioral cloning model on the given dataset."""
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
    states_continuous = data_file['states_continuous']
    states_discrete = data_file['states_discrete']
    actions = data_file['actions']
    rewards = data_file['rewards']
    terminated = data_file['terminated']

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
            name=experiment_name
        )
        params = {
            "nb_epochs": nb_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "test_ratio": test_ratio,
            **cfg
        }
        neptune_run["parameters"] = params

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create dataset
    dataset = TrajectoryDataset(states_continuous, states_discrete, actions, rewards, terminated, cfg=cfg)

    # Split into train and test sets
    train_dataset, test_dataset = dataset.train_test_split(test_ratio=test_ratio)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    state_dim = dataset.get_state_dim()
    action_dims = dataset.get_action_dims()
    model = ActorMLP(input_dim=state_dim, hidden_dim=cfg["hidden_dim"], output_dims=action_dims).to(device)

    # Training loop (omitted for brevity)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    tqdm_bar = tqdm(range(nb_epochs), desc=f"Epoch 0/{nb_epochs}")
    for epoch in tqdm_bar:
        # Training step
        model.train()
        train_accuracies = []
        for states_batch, actions_batch, _ in train_loader:
            states_batch = states_batch.to(device)
            actions_batch = actions_batch.to(device)

            optimizer.zero_grad()
            action_logits = model(states_batch)

            loss = 0.0
            for i, (logits, name) in enumerate(zip(action_logits, Action.NAMES)):
                action_loss = loss_fn(logits, actions_batch[:, i])
                loss += action_loss
                acc = (logits.argmax(dim=1) == actions_batch[:, i]).float().mean()
                train_accuracies.append(acc.item())
                if logging:
                    neptune_run[f'train/{name}_accuracy'].append(acc.item())
                    neptune_run[f'train/{name}_loss'].append(action_loss.item())
            if logging:
                neptune_run['train/total_loss'].append(loss.item())
            loss.backward()
            optimizer.step()

        # Evaluation step
        model.eval()
        test_accuracies = []
        with torch.no_grad():
            for states_batch, actions_batch, _ in test_loader:
                states_batch = states_batch.to(device)
                actions_batch = actions_batch.to(device)

                action_logits = model(states_batch)

                for i, (logits, name) in enumerate(zip(action_logits, Action.NAMES)):
                    action_loss = loss_fn(logits, actions_batch[:, i])
                    acc = (logits.argmax(dim=1) == actions_batch[:, i]).float().mean()
                    test_accuracies.append(acc.item())
                    if logging:
                        neptune_run[f'test/{name}_accuracy'].append(acc.item())
                        neptune_run[f'test/{name}_loss'].append(action_loss.item())

        # Save model
        torch.save(model.state_dict(), model_path)

        # Update progress bar description
        tqdm_bar.set_description(f"Epoch {epoch+1}/{nb_epochs} - Train Acc: {np.mean(train_accuracies).item():.2f} - Test Acc: {np.mean(test_accuracies).item():.2f}")

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
        nb_epochs=args.nb_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        test_ratio=args.test_ratio,
        logging=args.logging
    )
