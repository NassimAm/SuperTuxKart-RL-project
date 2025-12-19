import gymnasium as gym
from pystk2_gymnasium import AgentSpec
from pystk2_gymnasium.stk_wrappers import ConstantSizedObservations
from stk_actor.core.wrappers import EnhanceObservations, CustomPolarObservations, StuckObservations, ExpertObservations
from stk_actor.core.state import State
from stk_actor.core.actor import Actor
from tqdm import tqdm
import numpy as np
import argparse


def benchmark_nn(model_path: str, data_path: str, difficulty: int = 2, nb_races_per_track: int = 10, all_features: bool = False):
    """ Benchmark a trained neural network actor model and save the results to a file.

    Args:
        model_path (str): The path to the trained actor model.
        data_path (str): The path to save the benchmark results.
        difficulty (int): The difficulty level to benchmark on.
        nb_races_per_track (int): The number of races to run per track.
        all_features (bool): Whether the model uses all features or not.
    """
    if all_features:
        print("Benchmarking with all features enabled.")
    # Load the environment
    env = gym.make("supertuxkart/full-v0", render_mode=None, agent=AgentSpec(use_ai=False), difficulty=difficulty, num_kart=10)
    wrappers = [
        lambda env: ConstantSizedObservations(env, state_paths=10),
        lambda env: EnhanceObservations(env, previous_action_horizon=1),
        lambda env: StuckObservations(env),
        lambda env: ExpertObservations(env),
        lambda env: CustomPolarObservations(env)
    ]
    for wrapper in wrappers:
        env = wrapper(env)
    # Get list of tracks
    tracks = env.unwrapped.TRACKS
    # Load the actor model
    state, _ = env.reset()
    actor = Actor(model_path, State.to_model(State.from_env(state), all_features=all_features).shape[0], hidden_dim=256, head_dim=64)
    # Run multiple trials
    nb_trials = nb_races_per_track * len(tracks)
    tqdm_bar = tqdm(range(nb_trials), desc="Trials")
    positions_data = {
        'track': [],
        'position': [],
        'reward': []
    }
    for trial in tqdm_bar:
        # Get track_id
        track_id = trial // nb_races_per_track
        track = tracks[track_id]
        # Reset the environment
        env.unwrapped.default_track = track
        state, info = env.reset()
        positions_data['track'].append(track)
        reward = 0.0
        done = False
        nb_steps = 0
        # Run the episode
        while not done:
            # Get actor action
            action = actor.act(state, all_features=all_features)

            # Step the environment
            next_state, reward, terminated, truncated, info = env.step(
                action
            )

            state = next_state
            nb_steps += 1
            done = terminated or truncated
        # Record the final position
        positions_data['position'].append(info['position'])
        positions_data['reward'].append(reward)
        tqdm_bar.set_description(f"Trial {trial+1}/{nb_trials} - Position Avg: {np.mean(positions_data['position']).item():.2f}, Position Std: {np.std(positions_data['position']).item():.2f}, Reward Avg: {np.mean(positions_data['reward']).item():.2f}, Reward Std: {np.std(positions_data['reward']).item():.2f}")
    # Close the environment
    env.close()
    # Save the results in file
    np.savez(data_path, **positions_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a trained neural network actor model.")
    parser.add_argument("--model_path", "-m", type=str, required=True, help="Path to the trained actor model.")
    parser.add_argument("--data_path", "-d", type=str, required=True, help="Path to save the benchmark results.")
    parser.add_argument("--difficulty", type=int, default=2, help="Difficulty level to benchmark on.")
    parser.add_argument("--nb_races_per_track", type=int, default=10, help="Number of races to run per track for the benchmark.")
    parser.add_argument("--all_features", action='store_true', help="Whether the model uses all features or not.")
    args = parser.parse_args()

    benchmark_nn(
        model_path=args.model_path,
        data_path=args.data_path,
        difficulty=args.difficulty,
        nb_races_per_track=args.nb_races_per_track,
        all_features=args.all_features
    )
