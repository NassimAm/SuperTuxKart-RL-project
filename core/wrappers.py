from gymnasium import Env, Wrapper, ObservationWrapper
from gymnasium.spaces import Box, Discrete, Sequence
from core.state import Phase, Powerup, Attachment
from core.expert import ExpertAgent
from core.utils import from_cartesian_to_polar, get_xz_angle_between, rotate
from copy import deepcopy
import numpy as np
from typing import Optional


def kartview(env, x, make_relative_to_kart: bool = True, adapt_to_kart_rotation: bool = True):
    """Returns a vector in the kart frame

    X right, Y up, Z forwards
    """
    kart = env.unwrapped.world.karts[env.unwrapped.kart_ix]
    x_proj = x
    if make_relative_to_kart:
        x_proj = x - kart.location
    if adapt_to_kart_rotation:
        x_proj = rotate(x_proj, kart.rotation)
    return x_proj


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
        self.observation_space['should_accelerate_in_start'] = Discrete(
            n=2,
            dtype=np.int32,
        )
        # Update with the new powerup state
        self.observation_space['powerup'] = Discrete(
            n=len(Powerup),
            dtype=env.observation_space['powerup'].dtype,
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
            observation['should_accelerate_in_start'] = 0
        elif self.nb_steps < Phase.get_set_phase_limit():
            observation['phase'] = Phase.SET_PHASE.value
            if self.nb_steps >= Phase.get_ready_phase_limit() + 1:  # Delay by one step to increase start boost chances
                observation['should_accelerate_in_start'] = 1
            else:
                observation['should_accelerate_in_start'] = 0
        else:
            observation['phase'] = Phase.RACE_PHASE.value
            observation['should_accelerate_in_start'] = 1
        # Powerup representation
        powerup = self.env.unwrapped.world.karts[self.env.unwrapped.kart_ix].powerup.type.value
        observation['powerup'] = powerup
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


class ExpertObservations(Wrapper):
    """An observation wrapper that adds more detailed observations."""

    def __init__(self, env: Env, previous_action_horizon: int = 1):
        """An observation wrapper that adds more detailed observations.

        :param env: The environment to wrap.
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
        # Distance traveled down the track
        self.observation_space['distance_down_track'] = Box(
            low=0.0,
            high=1.0,
            shape=(),
            dtype=np.float32,
        )
        # Overall distance ratio in the race
        self.observation_space['overall_distance_ratio'] = Box(
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
        # Add is on road
        self.observation_space['is_on_road'] = Discrete(
            n=2,
            dtype=np.int32,
        )
        # Add nitro time left
        self.nitro_duration_time_left = 0.0
        self.observation_space['nitro_duration_time_left'] = Box(
            low=0.0,
            high=float('inf'),
            shape=(),
            dtype=np.float32,
        )
        self.nitro_fadeout_time_left = 0.0
        self.observation_space['nitro_fadeout_time_left'] = Box(
            low=0.0,
            high=float('inf'),
            shape=(),
            dtype=np.float32,
        )
        # Add zipper time left
        self.zipper_duration_time_left = 0.0
        self.observation_space['zipper_duration_time_left'] = Box(
            low=0.0,
            high=float('inf'),
            shape=(),
            dtype=np.float32,
        )
        self.zipper_fadeout_time_left = 0.0
        self.observation_space['zipper_fadeout_time_left'] = Box(
            low=0.0,
            high=float('inf'),
            shape=(),
            dtype=np.float32,
        )
        # Path widths
        self.observation_space['paths_width'] = Box(
            low=0.0,
            high=float('inf'),
            shape=(self.env.observation_space['paths_start'].shape[0],),
            dtype=np.float32,
        )
        # Path distances
        self.observation_space['paths_distance'] = Box(
            low=0.0,
            high=1.0,
            shape=(self.env.observation_space['paths_start'].shape[0],),
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
        # Add oppponent karts attachement
        self.observation_space['karts_attachment'] = Box(
            low=0,
            high=len(Attachment)-1,
            shape=(self.state_karts,),
            dtype=np.int32
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
        # Add time since last fire
        self.time_since_last_fire = 0.0
        self.observation_space['time_since_last_fire'] = Box(
            low=0.0,
            high=ExpertAgent.MAX_TIME_SINCE_LAST_FIRE,
            shape=(),
            dtype=np.float32,
        )
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
        # Add overall distance ratio
        overall_distance = max(0.0, kart.overall_distance)
        track_length = self.env.unwrapped.track.length
        full_distance = self.env.unwrapped.config.laps * track_length
        observation['distance_down_track'] = max(0.0, min(kart.distance_down_track / track_length, 1.0))
        observation['overall_distance_ratio'] = max(0.0, min(overall_distance / full_distance, 1.0))
        # Get more accurate paths
        # Start with the current path
        current_node = kart.node
        current_path_start = kartview(self.env, self.env.unwrapped.track.path_nodes[current_node][0])
        current_path_end = kartview(self.env, self.env.unwrapped.track.path_nodes[current_node][1])
        current_path_width = self.env.unwrapped.track.path_width[current_node]
        current_path_distance = self.env.unwrapped.track.path_distance[current_node]
        paths_start = [current_path_start]
        paths_end = [current_path_end]
        paths_width = [current_path_width]
        paths_distance = [current_path_distance]
        state_paths = observation['paths_start'].shape[0]
        for _ in range(1, state_paths):
            successor = self.env.unwrapped.track.successors[current_node][0]  # For simplicity, take the first successor
            path_start = kartview(self.env, self.env.unwrapped.track.path_nodes[successor][0])
            path_end = kartview(self.env, self.env.unwrapped.track.path_nodes[successor][1])
            path_width = self.env.unwrapped.track.path_width[successor]
            path_distance = self.env.unwrapped.track.path_distance[successor]
            paths_start.append(path_start)
            paths_end.append(path_end)
            paths_width.append(path_width)
            paths_distance.append(path_distance)
            current_node = successor
        # Convert to numpy arrays
        observation['paths_start'] = np.array(paths_start)
        observation['paths_end'] = np.array(paths_end)
        observation['paths_width'] = np.array(paths_width)
        observation['paths_distance'] = np.array(paths_distance)
        # Normalize path distances
        observation['paths_distance'] = np.clip(observation['paths_distance'][:, 1] / track_length, 0.0, 1.0)
        # Get distance to center of current path
        path_start = observation['paths_start'][0]
        path_end = observation['paths_end'][0]
        path_dir = path_end - path_start
        path_length = np.linalg.norm(path_dir)
        path_dir /= path_length
        rel_pos = -path_start
        proj_length = np.dot(rel_pos, path_dir)
        proj_h = rel_pos - proj_length * path_dir
        signed_distance = (np.sign(get_xz_angle_between(path_dir, proj_h)) * np.linalg.norm(proj_h)).item()
        observation['center_path_distance'] = signed_distance
        # Add current lap
        observation['current_lap'] = kart.finished_laps
        # Add game time
        observation['game_time'] = self.env.unwrapped.world.time
        # Add is on road
        observation['is_on_road'] = int(kart.is_on_road)
        # Add nitro time left
        if did_nitro:
            # If the kart used nitro, reset the nitro time left
            self.nitro_duration_time_left = ExpertAgent.NITRO_DURATION
            self.nitro_fadeout_time_left = ExpertAgent.NITRO_FADE_OUT_TIME
            observation['nitro_duration_time_left'] = self.nitro_duration_time_left
            observation['nitro_fadeout_time_left'] = self.nitro_fadeout_time_left
        else:
            if reset:
                # If resetting, set nitro time left to 0.0
                self.nitro_duration_time_left = 0.0
                self.nitro_fadeout_time_left = 0.0
                observation['nitro_duration_time_left'] = self.nitro_duration_time_left
                observation['nitro_fadeout_time_left'] = self.nitro_fadeout_time_left
            else:
                # Update duration and fadeout times
                nitro_time_left = self.nitro_duration_time_left + self.nitro_fadeout_time_left
                nitro_time_left = max(0.0, nitro_time_left - self.env.unwrapped.config.step_size)
                if nitro_time_left <= ExpertAgent.NITRO_FADE_OUT_TIME:
                    self.nitro_duration_time_left = 0.0
                    self.nitro_fadeout_time_left = nitro_time_left
                else:
                    self.nitro_duration_time_left = nitro_time_left - ExpertAgent.NITRO_FADE_OUT_TIME
                    self.nitro_fadeout_time_left = ExpertAgent.NITRO_FADE_OUT_TIME
                observation['nitro_duration_time_left'] = self.nitro_duration_time_left
                observation['nitro_fadeout_time_left'] = self.nitro_fadeout_time_left
        # Add zipper time left
        if did_zipper:
            # If the kart used zipper, reset the zipper time left
            self.zipper_duration_time_left = ExpertAgent.ZIPPER_DURATION
            self.zipper_fadeout_time_left = ExpertAgent.ZIPPER_FADE_OUT_TIME
            observation['zipper_duration_time_left'] = self.zipper_duration_time_left
            observation['zipper_fadeout_time_left'] = self.zipper_fadeout_time_left
        else:
            if reset:
                # If resetting, set zipper time left to 0.0
                self.zipper_duration_time_left = 0.0
                self.zipper_fadeout_time_left = 0.0
                observation['zipper_duration_time_left'] = self.zipper_duration_time_left
                observation['zipper_fadeout_time_left'] = self.zipper_fadeout_time_left
            else:
                # Decrease zipper time left
                zipper_time_left = self.zipper_duration_time_left + self.zipper_fadeout_time_left
                zipper_time_left = max(0.0, zipper_time_left - self.env.unwrapped.config.step_size)
                if zipper_time_left <= ExpertAgent.ZIPPER_FADE_OUT_TIME:
                    self.zipper_duration_time_left = 0.0
                    self.zipper_fadeout_time_left = zipper_time_left
                else:
                    self.zipper_duration_time_left = zipper_time_left - ExpertAgent.ZIPPER_FADE_OUT_TIME
                    self.zipper_fadeout_time_left = ExpertAgent.ZIPPER_FADE_OUT_TIME
                observation['zipper_duration_time_left'] = self.zipper_duration_time_left
                observation['zipper_fadeout_time_left'] = self.zipper_fadeout_time_left
        # Change kart positions, add opponent karts relative rank and velocity, all sorted by track distance difference
        kart_positions = []
        karts_rel_rank = []
        karts_velocity = []
        karts_track_diff = []
        karts_attachment = []
        for kart_ix, other_kart in enumerate(self.env.unwrapped.world.karts):
            if kart_ix == self.env.unwrapped.kart_ix:
                continue  # Skip self
            # Relative position
            kart_pos = kartview(self.env, other_kart.location)
            kart_positions.append(kart_pos)
            # Relative rank
            other_rank = other_kart.position
            normalized_other_rank = 1.0 - ((other_rank - 1) / (num_karts - 1))  # 0.0 for last, 1.0 for first and others in between
            rel_rank = normalized_other_rank - normalized_rank
            karts_rel_rank.append(rel_rank)
            # Velocity (3D vector)
            velocity = kartview(self.env, other_kart.velocity, make_relative_to_kart=False)
            karts_velocity.append(velocity)
            # Track distance difference
            other_kart_overall_distance = max(0.0, other_kart.overall_distance)
            track_diff = abs(other_kart_overall_distance - overall_distance)
            karts_track_diff.append(track_diff)
            # Attachment
            attachment = other_kart.attachment.type.value
            karts_attachment.append(attachment)
        # Convert to numpy arrays
        kart_positions = np.array(kart_positions)
        karts_rel_rank = np.array(karts_rel_rank)
        karts_velocity = np.array(karts_velocity)
        karts_track_diff = np.array(karts_track_diff)
        karts_attachment = np.array(karts_attachment)
        # Sort all by track distance difference
        sorted_indices = np.argsort(karts_track_diff)
        kart_positions = kart_positions[sorted_indices]
        karts_rel_rank = karts_rel_rank[sorted_indices]
        karts_velocity = karts_velocity[sorted_indices]
        karts_track_diff = karts_track_diff[sorted_indices]
        karts_attachment = karts_attachment[sorted_indices]
        # Set in observation
        observation['karts_position'] = kart_positions[:self.state_karts]
        observation['karts_rel_rank'] = karts_rel_rank[:self.state_karts]
        observation['karts_velocity'] = karts_velocity[:self.state_karts]
        observation['karts_distance'] = karts_track_diff[:self.state_karts]
        observation['karts_attachment'] = karts_attachment[:self.state_karts]
        # Add skid direction and time since last fire
        if reset or (action is None):
            self.skid_direction = 0
            self.time_since_skid_start = 0.0
            self.time_since_last_fire = 0.0
        elif action is not None:
            # Drifting logic
            if not action['drift']:
                self.skid_direction = 0
                self.time_since_skid_start = 0.0
            else:
                if len(self.prev_actions) > 0:
                    last_action = self.prev_actions[-1]
                    speed = np.linalg.norm(observation['velocity'])
                    if (not last_action['drift']) and (speed > ExpertAgent.SKID_MIN_SPEED):
                        self.skid_direction = -1 if action['steer'] < 0 else 1
                    if self.skid_direction != 0:
                        self.time_since_skid_start += self.env.unwrapped.config.step_size
            # Firing logic
            if action['fire']:
                self.time_since_last_fire = 0.0
            else:
                self.time_since_last_fire += self.env.unwrapped.config.step_size
                self.time_since_last_fire = min(self.time_since_last_fire, ExpertAgent.MAX_TIME_SINCE_LAST_FIRE)
        # Set in observation
        observation['skid_direction'] = self.skid_direction
        observation['time_since_skid_start'] = self.time_since_skid_start
        observation['time_since_last_fire'] = self.time_since_last_fire
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
        obs, info = self.env.reset(**kwargs)
        new_obs = self.get_new_observation(obs, action=None, did_nitro=False, did_zipper=False, reset=True)
        return new_obs, info

    def step(self, action):
        """Step the environment and add expert observations."""
        kart = self.env.unwrapped.world.karts[self.env.unwrapped.kart_ix]
        did_nitro = action['nitro']
        # TODO: Improve detection of zipper usage if possible
        did_zipper = (kart.powerup.type.value == Powerup.ZIPPER.value) and action['fire']
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_obs = self.get_new_observation(obs, action=action, did_nitro=did_nitro, did_zipper=did_zipper, reset=False)
        return new_obs, reward, terminated, truncated, info


class StuckObservationWrapper(Wrapper):
    """An observation wrapper that adds a 'stuck' indicator to the observation."""

    def __init__(self, env: Env):
        """An observation wrapper that adds a 'stuck' indicator to the observation.

        :param env: The environment to wrap.
        """
        super().__init__(env)

        # Update the observation space to include the 'stuck' indicator
        self.observation_space = deepcopy(env.observation_space)
        self.is_stuck = False
        self.is_being_rescued = False
        self.observation_space['stuck'] = Discrete(
            n=2,  # 0: not stuck, 1: stuck
            dtype=np.int32,
        )
        self.time_since_stuck = 0.0
        self.observation_space['time_since_stuck'] = Box(
            low=0.0,
            high=float('inf'),
            shape=(),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        """Reset the environment and the stuck indicators.

        :return: The initial observation and info dictionary.
        """
        observation, info = self.env.reset(**kwargs)
        self.is_stuck = False
        self.is_being_rescued = False
        self.time_since_stuck = 0.0
        observation['stuck'] = int(self.is_stuck)
        observation['time_since_stuck'] = self.time_since_stuck
        return observation, info

    def step(self, action):
        """Step the environment and update the stuck indicators.

        :param action: The action to take.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = self.get_new_observation(observation)
        return observation, reward, terminated, truncated, info

    def get_new_observation(self, observation):
        """Add the 'stuck' indicator to the observation.

        :param observation: The original observation.
        :return: The modified observation with the 'stuck' indicator.
        """
        # Determine if the kart is being rescued
        path_start = observation['paths_start'][0]
        path_end = observation['paths_end'][0]
        path_width = observation['paths_width'][0][0]
        a = (path_end[1] - path_start[1]) / (path_end[2] - path_start[2])
        b = path_start[1] - a * path_start[2]
        distance_above_track = abs(min(b, 0)) - ExpertAgent.KART_HEIGHT / 2.0
        if not self.is_being_rescued:  # If not already being rescued, check conditions to become rescued
            self.is_being_rescued = ((not observation['jumping']) and  # Not jumping
                                     (abs(path_start[0]) < path_width / 2.0) and  # Within the path
                                     (abs(path_end[0]) < path_width / 2.0) and  # Within the path
                                     (distance_above_track > 0.5))  # Above the track by more than half a meter
        else:  # If already being rescued, check conditions to become not rescued
            self.is_being_rescued = not ((distance_above_track < 0.05) and
                                         (observation['prev_actions'][-1]['acceleration'] > 0) and
                                         (observation['velocity'][2] > 0.2))  # Negation of "Is on track, tried to accelerate and succeeded"
        # Determine if the kart is stuck
        if not self.is_stuck:  # If not already stuck, check conditions to become stuck
            self.is_stuck = int((observation['phase'] == Phase.RACE_PHASE.value) and  # In race phase
                                (not observation['jumping']) and  # Not jumping
                                (not self.is_being_rescued) and  # Not being rescued
                                (observation['prev_actions'][-1]['acceleration'] > 0) and  # Trying to accelerate
                                (observation['velocity'][2] < 0.2))  # Very low forward velocity
        else:  # If already stuck, check conditions to become not stuck
            self.is_stuck = int(((not self.is_being_rescued) and  # Not being rescued
                                (not ((observation['prev_actions'][-1]['acceleration'] > 0) and
                                      (observation['velocity'][2] > 0.2)))) or
                                (self.time_since_stuck < 2.0))  # Negation of "Tried to accelerate and succeeded"
        # Update time since stuck
        if self.is_stuck:
            self.time_since_stuck += self.env.unwrapped.config.step_size
            self.time_since_stuck = min(self.time_since_stuck, 5.0)  # Cap at 5 seconds
        else:
            self.time_since_stuck = 0.0
        # Update the observation
        observation['stuck'] = self.is_stuck
        observation['time_since_stuck'] = self.time_since_stuck
        return observation


class CustomPolarObservations(ObservationWrapper):
    """Modifies position to polar positions

    Angles are in radian

    input: X right, Y up, Z forwards
    output: (angle in the ZX plane, angle in the ZY plane, distance)
    """
    # TODO: Recheck what features should be in polar coordinates (for example velocities)
    #: Keys to transform (batch)
    KEYS = ["items_position", "karts_position", "paths_start", "paths_end", "karts_velocity"]
    #: Keys to transform (single)
    SIMPLE_KEYS = ["center_path", "velocity"]

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
