import gymnasium as gym
from pystk2_gymnasium import AgentSpec
from pystk2_gymnasium.stk_wrappers import ConstantSizedObservations
from core.wrappers import EnhanceObservations, CustomPolarObservations, StuckObservations, ExpertObservations
from core.state import State
from core.actor import Actor
from tqdm import tqdm
import numpy as np
import argparse


def benchmark_nn(model_path: str, data_path: str, difficulty: int = 2, nb_races: int = 100):
    # Load the environment
    env = gym.make("supertuxkart/full-v0", render_mode=None, agent=AgentSpec(use_ai=False), difficulty=difficulty, num_kart=10)
    wrappers = [
        lambda env: ConstantSizedObservations(env, state_paths=10),
        lambda env: EnhanceObservations(env),
        lambda env: StuckObservations(env),
        lambda env: ExpertObservations(env),
        lambda env: CustomPolarObservations(env)
    ]
    for wrapper in wrappers:
        env = wrapper(env)
    # Load the actor model
    state, _ = env.reset()
    actor = Actor(model_path, State.to_model(State.from_env(state)).shape[0], 512)
    # Run multiple trials
    nb_trials = nb_races
    tqdm_bar = tqdm(range(nb_trials), desc="Trials")
    positions_data = {
        'track': [],
        'position': [],
        'reward_mean': [],
        'reward_std': []
    }
    for trial in tqdm_bar:
        # Reset the environment
        state, info = env.reset()
        positions_data['track'].append(env.unwrapped.current_track)
        rewards = []
        done = False
        nb_steps = 0
        # Run the episode
        while not done:
            # Get actor action
            action = actor.act(state)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a trained neural network actor model.")
    parser.add_argument("--model_path", "-m", type=str, required=True, help="Path to the trained actor model.")
    parser.add_argument("--data_path", "-d", type=str, required=True, help="Path to save the benchmark results.")
    parser.add_argument("--difficulty", type=int, default=2, help="Difficulty level to benchmark on.")
    parser.add_argument("--nb_races", type=int, default=100, help="Number of races to run for the benchmark.")
    args = parser.parse_args()

    benchmark_nn(
        model_path=args.model_path,
        data_path=args.data_path,
        difficulty=args.difficulty,
        nb_races=args.nb_races
    )
