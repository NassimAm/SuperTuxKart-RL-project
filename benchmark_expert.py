import gymnasium as gym
from pystk2_gymnasium import AgentSpec
from pystk2_gymnasium.stk_wrappers import ConstantSizedObservations
from stk_actor.core.wrappers import EnhanceObservations1, EnhanceObservations2, CustomPolarObservations, StuckObservations, ExpertObservations
from stk_actor.core.expert import ExpertAgent
import numpy as np
from tqdm import tqdm
import argparse


def benchmark_expert_on_difficulty(difficulty: int, data_path: str, enable_drift: bool = True, nb_races_per_track: int = 10):
    """Benchmark the expert agent on a given difficulty level and save the results to a file.

    Args:
        difficulty (int): The difficulty level to benchmark on.
        data_path (str): The path to save the benchmark results.
        enable_drift (bool): Whether to enable drift in the expert agent.
        nb_races_per_track (int): The number of races to run per track.
    """
    # Load the environment
    env = gym.make("supertuxkart/full-v0", render_mode=None, agent=AgentSpec(use_ai=False), difficulty=difficulty, num_kart=10)
    wrappers = [
        lambda env: EnhanceObservations1(env),
        lambda env: ConstantSizedObservations(env, state_paths=10),
        lambda env: EnhanceObservations2(env, previous_action_horizon=1),
        lambda env: StuckObservations(env),
        lambda env: ExpertObservations(env, avoid_karts=False),
        lambda env: CustomPolarObservations(env)
    ]
    for wrapper in wrappers:
        env = wrapper(env)
    # Get list of tracks
    tracks = env.unwrapped.TRACKS
    # Create the expert agent
    expert = ExpertAgent(difficulty=difficulty, enable_drift=enable_drift, avoid_karts=False)
    # Run multiple trials
    nb_trials = nb_races_per_track * len(tracks)
    tqdm_bar = tqdm(range(nb_trials), desc="Trials")
    positions_data = {
        'track': [],
        'position': [],
        'reward_mean': [],
        'reward_std': []
    }
    for trial in tqdm_bar:
        # Get track_id
        track_id = trial // nb_races_per_track
        track = tracks[track_id]
        # Reset the environment
        env.unwrapped.default_track = track
        state, info = env.reset()
        positions_data['track'].append(track)
        rewards = []
        done = False
        nb_steps = 0
        # Run the episode
        while not done:
            # Get expert action
            action = expert.act(state)

            # Step the environment
            next_state, reward, terminated, truncated, info = env.step(
                action
            )

            state = next_state
            rewards.append(reward)
            nb_steps += 1
            done = terminated or truncated
        # Record the final position
        positions_data['position'].append(info['position'])
        positions_data['reward_mean'].append(np.mean(rewards).item())
        positions_data['reward_std'].append(np.std(rewards).item())
        tqdm_bar.set_description(f"Trial {trial+1}/{nb_trials} - Position Avg: {np.mean(positions_data['position']).item():.2f}, Position Std: {np.std(positions_data['position']).item():.2f}")
    # Close the environment
    env.close()
    # Save the results in file
    np.savez(data_path, **positions_data)


def benchmark_expert(data_folder: str, nb_races_per_track: int = 10):
    """Benchmark the expert agent with its two variants (drift enabled and disabled) across multiple difficulty levels and save the results.

    Args:
        data_folder (str): The folder to save the benchmark results
        nb_races_per_track (int): The number of races to run per track.
    """
    # ======================= Benchmark with drift disabled =======================
    print("Benchmarking expert with drift disabled...")
    enable_drift = False
    # Difficulty 0 (Easy)
    print("Benchmarking expert on difficulty 0 (Easy)...")
    easy_difficulty = 0
    easy_difficulty_path = f"{data_folder}/expert_without_drift_difficulty_{easy_difficulty}.npz"
    benchmark_expert_on_difficulty(
        difficulty=easy_difficulty,
        data_path=easy_difficulty_path,
        enable_drift=enable_drift,
        nb_races_per_track=nb_races_per_track
    )
    # Difficulty 1 (Medium)
    print("Benchmarking expert on difficulty 1 (Medium)...")
    medium_difficulty = 1
    medium_difficulty_path = f"{data_folder}/expert_without_drift_difficulty_{medium_difficulty}.npz"
    benchmark_expert_on_difficulty(
        difficulty=medium_difficulty,
        data_path=medium_difficulty_path,
        enable_drift=enable_drift,
        nb_races_per_track=nb_races_per_track
    )
    # Difficulty 2 (Hard)
    print("Benchmarking expert on difficulty 2 (Hard)...")
    hard_difficulty = 2
    hard_difficulty_path = f"{data_folder}/expert_without_drift_difficulty_{hard_difficulty}.npz"
    benchmark_expert_on_difficulty(
        difficulty=hard_difficulty,
        data_path=hard_difficulty_path,
        enable_drift=enable_drift,
        nb_races_per_track=nb_races_per_track
    )
    # Difficulty 3 (SuperTux)
    print("Benchmarking expert on difficulty 3 (SuperTux)...")
    supertux_difficulty = 3
    supertux_difficulty_path = f"{data_folder}/expert_without_drift_difficulty_{supertux_difficulty}.npz"
    benchmark_expert_on_difficulty(
        difficulty=supertux_difficulty,
        data_path=supertux_difficulty_path,
        enable_drift=enable_drift,
        nb_races_per_track=nb_races_per_track
    )
    # ======================= Benchmark with drift enabled =======================
    print("Benchmarking expert with drift enabled...")
    enable_drift = True
    # Difficulty 0 (Easy)
    print("Benchmarking expert on difficulty 0 (Easy)...")
    easy_difficulty = 0
    easy_difficulty_path = f"{data_folder}/expert_with_drift_difficulty_{easy_difficulty}.npz"
    benchmark_expert_on_difficulty(
        difficulty=easy_difficulty,
        data_path=easy_difficulty_path,
        enable_drift=enable_drift,
        nb_races_per_track=nb_races_per_track
    )
    # Difficulty 1 (Medium)
    print("Benchmarking expert on difficulty 1 (Medium)...")
    medium_difficulty = 1
    medium_difficulty_path = f"{data_folder}/expert_with_drift_difficulty_{medium_difficulty}.npz"
    benchmark_expert_on_difficulty(
        difficulty=medium_difficulty,
        data_path=medium_difficulty_path,
        enable_drift=enable_drift,
        nb_races_per_track=nb_races_per_track
    )
    # Difficulty 2 (Hard)
    print("Benchmarking expert on difficulty 2 (Hard)...")
    hard_difficulty = 2
    hard_difficulty_path = f"{data_folder}/expert_with_drift_difficulty_{hard_difficulty}.npz"
    benchmark_expert_on_difficulty(
        difficulty=hard_difficulty,
        data_path=hard_difficulty_path,
        enable_drift=enable_drift,
        nb_races_per_track=nb_races_per_track
    )
    # Difficulty 3 (SuperTux)
    print("Benchmarking expert on difficulty 3 (SuperTux)...")
    supertux_difficulty = 3
    supertux_difficulty_path = f"{data_folder}/expert_with_drift_difficulty_{supertux_difficulty}.npz"
    benchmark_expert_on_difficulty(
        difficulty=supertux_difficulty,
        data_path=supertux_difficulty_path,
        enable_drift=enable_drift,
        nb_races_per_track=nb_races_per_track
    )
    # Done
    print("Benchmarking completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Expert Agent")
    parser.add_argument(
        "--data-folder",
        "-d",
        type=str,
        default="data",
        help="Folder to save the benchmark results"
    )
    parser.add_argument(
        "--nb-races-per-track",
        "-n",
        type=int,
        default=10,
        help="Number of races to run per track"
    )
    args = parser.parse_args()

    benchmark_expert(
        data_folder=args.data_folder,
        nb_races_per_track=args.nb_races_per_track
    )
