from gymnasium import Env, Wrapper, ObservationWrapper
from gymnasium.spaces import Box, Discrete, Sequence
from core.state import Phase, Powerup
from core.expert import ExpertAgent
from core.utils import from_cartesian_to_polar, rotate
from copy import deepcopy
import numpy as np
from typing import Optional


class CustomPolarObservations(ObservationWrapper):
    """Modifies position to polar positions

    Angles are in radian

    input: X right, Y up, Z forwards
    output: (angle in the ZX plane, angle in the ZY plane, distance)
    """
    # TODO: Recheck what features should be in polar coordinates (for example velocities)
    #: Keys to transform (batch)
    KEYS = ["items_position", "karts_position", "paths_start", "paths_end"]
    #: Keys to transform (single)
    SIMPLE_KEYS = ["center_path"]

    def __init__(self, env: Env, **kwargs):
        super().__init__(env, **kwargs)

    def observation(self, obs):
        # Shallow copy
        obs = {**obs}

        for key in CustomPolarObservations.SIMPLE_KEYS:
            v = obs[key]
            v[:] = from_cartesian_to_polar(v)

        for key in CustomPolarObservations.KEYS:
            v = obs[key]

            is_tuple = False
            if isinstance(v, tuple):
                is_tuple = True
                v = np.stack(v)
            v[:] = from_cartesian_to_polar(v)

            if is_tuple:
                obs[key] = tuple(x for x in v)

        return obs


class EnhanceObservations(Wrapper):
    """An observation wrapper that enhances the observation by correcting missing values and removing irrelevant data."""

    def __init__(self, env: Env):
        """An observation wrapper that enhances the observation by correcting missing values and removing irrelevant data.

        :param env: The environment to wrap.
        """
        super().__init__(env)

        self.observation_space = deepcopy(env.observation_space)
        # Update the observation space to include the new phase representation
        self.observation_space['phase'] = Discrete(
            n=len(Phase),
            dtype=env.observation_space['phase'].dtype,
        )
        self.observation_space['nb_steps_since_set'] = Box(
            low=0,
            high=float('inf'),
            shape=(),
            dtype=np.int32,
        )
        # Save number of steps to track phases
        self.nb_steps = 0

    def enhance_observation(self, observation):
        """Enhance the observation with better data.

        :param observation: The original observation.
        :return: The modified observation with the correct phase representation.
        """
        # Phase representation
        if self.nb_steps < Phase.get_ready_phase_limit():
            observation['phase'] = Phase.READY_PHASE.value
            observation['nb_steps_since_set'] = 0
        elif self.nb_steps < Phase.get_set_phase_limit():
            observation['phase'] = Phase.SET_PHASE.value
            observation['nb_steps_since_set'] = self.nb_steps - Phase.get_ready_phase_limit()
        else:
            observation['phase'] = Phase.RACE_PHASE.value
            observation['nb_steps_since_set'] = Phase.get_set_phase_limit() - Phase.get_ready_phase_limit()
        return observation

    def reset(self, **kwargs):
        """Reset the environment and the step counter.

        :return: The initial observation and info dictionary.
        """
        observation, info = self.env.reset(**kwargs)
        self.nb_steps = 0
        observation = self.enhance_observation(observation)
        return observation, info

    def step(self, action):
        """Step the environment and update the step counter.

        :param action: The action to take.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.nb_steps += 1
        observation = self.enhance_observation(observation)

        return observation, reward, terminated, truncated, info


class StuckObservationWrapper(ObservationWrapper):
    """An observation wrapper that adds a 'stuck' indicator to the observation."""

    def __init__(self, env: Env):
        """An observation wrapper that adds a 'stuck' indicator to the observation.

        :param env: The environment to wrap.
        """
        super().__init__(env)

        # Update the observation space to include the 'stuck' indicator
        self.observation_space = deepcopy(env.observation_space)
        self.observation_space['stuck'] = Discrete(
            n=2,  # 0: not stuck, 1: stuck
            dtype=np.int32,
        )

    def observation(self, observation):
        """Add the 'stuck' indicator to the observation.

        :param observation: The original observation.
        :return: The modified observation with the 'stuck' indicator.
        """
        observation['stuck'] = int((observation['phase'] == Phase.RACE_PHASE.value) and (np.linalg.norm(observation['velocity']) < 0.2))  # TODO: This isn't very robust, correct it
        return observation


class FeatureSelectionWrapper(ObservationWrapper):
    """An observation wrapper that selects specific features from the observation."""

    def __init__(self, env: Env, features_to_remove: list[str]):
        """An observation wrapper that selects specific features from the observation.

        :param env: The environment to wrap.
        :param features_to_remove: List of feature names to remove from the observation.
        """
        super().__init__(env)

        # Update the observation space to remove specified features
        self.observation_space = deepcopy(env.observation_space)
        for feature in features_to_remove:
            if feature in self.observation_space.spaces:
                del self.observation_space.spaces[feature]

    def observation(self, observation):
        """Remove specified features from the observation.

        :param observation: The original observation.
        :return: The modified observation with specified features removed.
        """
        for feature in observation.keys():
            if feature not in self.observation_space.spaces:
                del observation[feature]
        return observation


class ExpertObservations(Wrapper):
    """An observation wrapper that adds more detailed observations."""

    def __init__(self, env: Env, polar_observations: bool = True, previous_action_horizon: int = 1):
        """An observation wrapper that adds more detailed observations.

        :param env: The environment to wrap.
        :param polar_observations: Whether to use polar observations for karts and items.
        :param previous_action_horizon: The number of previous actions to store in the observation.
        """
        super().__init__(env)

        # Update the observation space to include the correct phase representation
        self.observation_space = deepcopy(env.observation_space)
        # Normalized kart position
        self.observation_space['rank'] = Box(
            low=0.0,
            high=1.0,
            shape=(),
            dtype=np.float32,
        )
        # Current lap number
        self.observation_space['current_lap'] = Box(
            low=0,
            high=float('inf'),
            shape=(),
            dtype=np.int32,
        )
        # Add game time
        self.observation_space['game_time'] = Box(
            low=0.0,
            high=float('inf'),
            shape=(),
            dtype=np.float32,
        )
        # Add nitro time left
        self.nitro_time_left = 0.0
        self.observation_space['nitro_time_left'] = Box(
            low=0.0,
            high=float('inf'),
            shape=(),
            dtype=np.float32,
        )
        # Add zipper time left
        self.zipper_time_left = 0.0
        self.observation_space['zipper_time_left'] = Box(
            low=0.0,
            high=float('inf'),
            shape=(),
            dtype=np.float32,
        )
        # Add opponent kart relative rank
        self.state_karts = self.env.observation_space['karts_position'].shape[0]
        self.observation_space['karts_rel_rank'] = Box(
            low=0.0,
            high=1.0,
            shape=(self.state_karts,),
            dtype=np.float32,
        )
        # Add opponent karts velocity
        self.observation_space['karts_velocity'] = Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(self.state_karts, 3),
            dtype=np.float32,
        )
        # Add opponent karts distance
        self.observation_space['karts_distance'] = Box(
            low=0.0,
            high=float('inf'),
            shape=(self.state_karts,),
            dtype=np.float32,
        )
        # Add skid direction
        self.skid_direction = 0
        self.observation_space['skid_direction'] = Box(
            low=-1,
            high=1,
            shape=(),
            dtype=np.int32,
        )
        # Add time since skidding started
        self.time_since_skid_start = 0.0
        self.observation_space['time_since_skid_start'] = Box(
            low=0.0,
            high=float('inf'),
            shape=(),
            dtype=np.float32,
        )
        # Save polar observations flag
        self.polar_observations = polar_observations
        # Get previous actions
        self.observation_space['prev_actions'] = Sequence(
            space=self.env.action_space
        )
        self.prev_actions = []
        self.previous_action_horizon = previous_action_horizon

    def get_new_observation(self, observation, action: Optional[dict] = None, did_nitro: bool = False, did_zipper: bool = False, reset: bool = False):
        """Get a new observation with expert observations added.

        :param observation: The original observation.
        :return: The modified observation with expert observations.
        """
        # Shallow copy
        observation = {**observation}
        # Get kart
        kart = self.env.unwrapped.world.karts[self.env.unwrapped.kart_ix]
        # Get number of karts
        num_karts = self.env.unwrapped.num_kart
        # Get unormalized kart rank
        rank = kart.position
        normalized_rank = 1.0 - ((rank - 1) / (num_karts - 1))  # 0.0 for last, 1.0 for first and others in between
        # Add normalized kart rank
        observation['rank'] = normalized_rank
        # Add current lap
        observation['current_lap'] = kart.finished_laps
        # Add game time
        observation['game_time'] = self.env.unwrapped.world.time
        # Add nitro time left
        if did_nitro:
            # If the kart used nitro, reset the nitro time left
            self.nitro_time_left = ExpertAgent.NITRO_DURATION + ExpertAgent.NITRO_FADE_OUT_TIME
            observation['nitro_time_left'] = self.nitro_time_left
        else:
            if reset:
                # If resetting, set nitro time left to 0.0
                self.nitro_time_left = 0.0
                observation['nitro_time_left'] = self.nitro_time_left
            else:
                # Decrease nitro time left
                new_nitro_time_left = max(0.0, self.nitro_time_left - self.env.unwrapped.config.step_size)
                self.nitro_time_left = new_nitro_time_left
                observation['nitro_time_left'] = new_nitro_time_left
        # Add zipper time left
        if did_zipper:
            # If the kart used zipper, reset the zipper time left
            self.zipper_time_left = ExpertAgent.ZIPPER_DURATION + ExpertAgent.ZIPPER_FADE_OUT_TIME
            observation['zipper_time_left'] = self.zipper_time_left
        else:
            if reset:
                # If resetting, set zipper time left to 0.0
                self.zipper_time_left = 0.0
                observation['zipper_time_left'] = self.zipper_time_left
            else:
                # Decrease zipper time left
                new_zipper_time_left = max(0.0, self.zipper_time_left - self.env.unwrapped.config.step_size)
                self.zipper_time_left = new_zipper_time_left
                observation['zipper_time_left'] = new_zipper_time_left
        # Change kart positions, add opponent karts relative rank and velocity, all sorted by track distance difference
        kart_positions = []
        karts_rel_rank = []
        karts_velocity = []
        karts_track_diff = []
        overall_distance = kart.overall_distance
        for kart_ix, other_kart in enumerate(self.env.unwrapped.world.karts):
            if kart_ix == self.env.unwrapped.kart_ix:
                continue  # Skip self
            # Relative position
            kart_pos = self.__kartview(other_kart.location)
            if self.polar_observations:
                kart_pos = from_cartesian_to_polar(kart_pos)
            kart_positions.append(kart_pos)
            # Relative rank
            other_rank = other_kart.position
            normalized_other_rank = 1.0 - ((other_rank - 1) / (num_karts - 1))  # 0.0 for last, 1.0 for first and others in between
            rel_rank = normalized_other_rank - normalized_rank
            karts_rel_rank.append(rel_rank)
            # Velocity (3D vector)
            velocity = self.__kartview(other_kart.velocity)
            karts_velocity.append(velocity)
            # Track distance difference
            track_diff = abs(other_kart.overall_distance - overall_distance)
            karts_track_diff.append(track_diff)
        # Convert to numpy arrays
        kart_positions = np.array(kart_positions)
        karts_rel_rank = np.array(karts_rel_rank)
        karts_velocity = np.array(karts_velocity)
        karts_track_diff = np.array(karts_track_diff)
        # Sort all by track distance difference
        sorted_indices = np.argsort(karts_track_diff)
        kart_positions = kart_positions[sorted_indices]
        karts_rel_rank = karts_rel_rank[sorted_indices]
        karts_velocity = karts_velocity[sorted_indices]
        karts_track_diff = karts_track_diff[sorted_indices]
        # Set in observation
        observation['karts_position'] = kart_positions[:self.state_karts]
        observation['karts_rel_rank'] = karts_rel_rank[:self.state_karts]
        observation['karts_velocity'] = karts_velocity[:self.state_karts]
        observation['karts_distance'] = karts_track_diff[:self.state_karts]
        # Add skid direction
        if reset or (action is None):
            self.skid_direction = 0
            self.time_since_skid_start = 0.0
        elif action is not None:
            if not action['drift']:
                self.skid_direction = 0
                self.time_since_skid_start = 0.0
            else:
                last_action = self.prev_actions[-1]
                speed = np.linalg.norm(observation['velocity'])
                if (not last_action['drift']) and (speed > ExpertAgent.SKID_MIN_SPEED):
                    self.skid_direction = -1 if action['steer'] < 0 else 1
                if self.skid_direction != 0:
                    self.time_since_skid_start += self.env.unwrapped.config.step_size
        observation['skid_direction'] = self.skid_direction
        observation['time_since_skid_start'] = self.time_since_skid_start
        # Add previous actions
        if reset:
            self.prev_actions = []
        else:
            if action is not None:
                if len(self.prev_actions) >= self.previous_action_horizon:
                    self.prev_actions.pop(0)  # Remove oldest
                self.prev_actions.append(action)
        observation['prev_actions'] = deepcopy(self.prev_actions)
        # Return modified observation
        return observation

    def reset(self, **kwargs):
        """Reset the environment and add expert observations."""
        self.nitro_time_left = 0.0
        self.zipper_time_left = 0.0
        obs, info = self.env.reset(**kwargs)
        new_obs = self.get_new_observation(obs, action=None, did_nitro=False, did_zipper=False, reset=True)
        return new_obs, info

    def step(self, action):
        """Step the environment and add expert observations."""
        kart = self.env.unwrapped.world.karts[self.env.unwrapped.kart_ix]
        did_nitro = action['nitro']
        # TODO: Improve detection of zipper usage
        did_zipper = (kart.powerup == Powerup.ZIPPER) and action['fire']
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_obs = self.get_new_observation(obs, action=action, did_nitro=did_nitro, did_zipper=did_zipper, reset=False)
        return new_obs, reward, terminated, truncated, info

    def __kartview(self, x):
        """Returns a vector in the kart frame

        X right, Y up, Z forwards
        """
        kart = self.env.unwrapped.world.karts[self.env.unwrapped.kart_ix]
        return rotate(x - kart.location, kart.rotation)

# class EnableFireBackwardsWrapper(ActionWrapper):
#     """An action wrapper that enables firing backwards."""

#     def __init__(self, env: Env):
#         """An action wrapper that enables firing backwards.

#         :param env: The environment to wrap.
#         """
#         super().__init__(env)
#         # Update the action space to include the look_back action
#         self.action_space = deepcopy(env.action_space)
#         self.action_space['fire'] = Discrete(
#             n=3,  # 0: no fire, 1: fire forwards, 2: fire backwards
#             dtype=env.action_space['fire'].dtype
#         )

#     def action(self, action):
#         """Convert the action to the environment's expected format.

#         :param action: The action to convert.
#         :return: The converted action.
#         """
#         env_action = deepcopy(action)

#         return env_action
