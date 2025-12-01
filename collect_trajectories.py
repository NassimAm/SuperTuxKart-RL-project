"""Using this script, we collect trajectories from the STK Bot at its max difficulty `difficulty = 2`
to pretrain our RL Agent."""

import gymnasium as gym
from pystk2_gymnasium import AgentSpec
from pystk2_gymnasium.stk_wrappers import ConstantSizedObservations
from core.state import State
from core.action import Action
from core.wrappers import EnhanceObservations, CustomPolarObservations, StuckObservationWrapper, ExpertObservations
from tqdm import tqdm
import numpy as np
import argparse


def collect_trajectories(data_path: str, difficulty: int = 3):
    """"""
    env = gym.make("supertuxkart/full-v0", render_mode=None, agent=AgentSpec(use_ai=True), difficulty=difficulty, num_kart=10, laps=3)
    previous_action_horizon = 1
    wrappers = [
        lambda env: ConstantSizedObservations(env, state_paths=10),
        lambda env: EnhanceObservations(env),
        lambda env: StuckObservationWrapper(env),
        lambda env: CustomPolarObservations(env),
        lambda env: ExpertObservations(env, polar_observations=True, previous_action_horizon=previous_action_horizon)
    ]
    for wrapper in wrappers:
        env = wrapper(env)

    state, _ = env.reset()
    nb_steps = 0
    rounds = 0

    states = []
    actions = []
    rewards = []
    terminateds = []
    truncateds = []

    tqdm_steps = tqdm(range(1, 1000001))
    for i in tqdm_steps:
        # Bot plays an action
        next_state, reward, terminated, truncated, _ = env.step(
            env.action_space.sample()
        )
        # Get that action
        action = next_state['action']
        done = terminated or truncated
        nb_steps += 1

        # Store data
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        terminateds.append(terminated)
        truncateds.append(truncated)

        # Save locally every 10000 steps
        if i % 10000 == 0 and i > 0:
            state_array = np.concat([State.to_model(State.from_env(s), previous_action_horizon=previous_action_horizon).numpy().reshape(1, -1) for s in states])
            actions_array = np.concat([Action.to_model(Action.from_env(a)).numpy().reshape(1, -1) for a in actions])
            rewards_array = np.array(rewards)
            terminateds_array = np.array(terminateds)
            truncateds_array = np.array(truncateds)
            data = {
                'states': state_array,
                'actions': actions_array,
                'rewards': rewards_array,
                'terminated': terminateds_array,
                'truncated': truncateds_array
            }
            np.savez_compressed(data_path, **data)

        # If done, reset the environment
        if done:
            next_state, _ = env.reset()
            rounds += 1
        # Update state
        state = next_state
        # Update progress bar
        tqdm_steps.set_description(f'size={nb_steps} rounds={rounds}')

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        default="data/train_data.npz",
        help="Path to save the collected trajectories.",
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        default=3,
        help="Difficulty level of the STK Bot.",
    )
    args = parser.parse_args()
    collect_trajectories(args.data_path, difficulty=args.difficulty)
