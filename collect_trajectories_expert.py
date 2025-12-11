"""Using this script, we collect trajectories from the STK Bot at its max difficulty `difficulty = 2`
to pretrain our RL Agent."""

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from pystk2_gymnasium import AgentSpec
from pystk2_gymnasium.stk_wrappers import ConstantSizedObservations
from core.state import State
from core.action import Action
from core.wrappers import EnhanceObservations, CustomPolarObservations, StuckObservations, ExpertObservations
from core.expert import ExpertAgent
from tqdm import tqdm
import numpy as np
import argparse


def collect_trajectories(data_path: str, enable_drift: bool = True, avoid_bad_items: bool = True,
                         avoid_karts: bool = True, use_nitro: bool = True, use_items: bool = True,
                         difficulty: int = 2):
    """"""
    env = gym.make("supertuxkart/full-v0", render_mode=None, agent=AgentSpec(use_ai=False), difficulty=difficulty, num_kart=10)
    previous_action_horizon = 1
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

    expert = ExpertAgent(env, enable_drift=enable_drift, avoid_bad_items=avoid_bad_items,
                         avoid_karts=avoid_karts, use_nitro=use_nitro, use_items=use_items)

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
        # Get expert action
        action = expert.act(state)
        # Play action in the environment
        next_state, reward, terminated, truncated, _ = env.step(
            action
        )
        done = terminated or truncated
        nb_steps += 1

        # Store data
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        terminateds.append(terminated)
        truncateds.append(truncated)

        # Save locally every 10000 steps
        if i % 100000 == 0 and i > 0:
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
            # Save data to a compressed .npz file with a step-based suffix
            path_without_extension, extension = data_path.split('.', 1)
            np.savez_compressed(f"{path_without_extension}_{i // 100000}.{extension}", **data)
            # Clear stored data after saving
            states.clear()
            actions.clear()
            rewards.clear()
            terminateds.clear()
            truncateds.clear()

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
        "--no_drift",
        action="store_true",
        help="Disable drifting behavior for the expert agent.",
    )
    parser.add_argument(
        "--no_avoid_bad_items",
        action="store_true",
        help="Disable avoidance of bad items for the expert agent.",
    )
    parser.add_argument(
        "--no_avoid_karts",
        action="store_true",
        help="Disable avoidance of other karts for the expert agent.",
    )
    parser.add_argument(
        "--no_use_nitro",
        action="store_true",
        help="Disable use of nitro for the expert agent.",
    )
    parser.add_argument(
        "--no_use_items",
        action="store_true",
        help="Disable use of items for the expert agent.",
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        default=2,
        help="Difficulty level of the STK Bot.",
    )
    args = parser.parse_args()
    collect_trajectories(args.data_path, enable_drift=not args.no_drift,
                         avoid_bad_items=not args.no_avoid_bad_items,
                         avoid_karts=not args.no_avoid_karts,
                         use_nitro=not args.no_use_nitro,
                         use_items=not args.no_use_items,
                         difficulty=args.difficulty)
