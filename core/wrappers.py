from gymnasium import Env, Wrapper, ObservationWrapper
from gymnasium.spaces import Box, Discrete, Sequence
from core.state import Phase, Powerup, Attachment, DirectionType, ItemType
from core.expert import ExpertAgent
from core.utils import from_cartesian_to_polar, get_xz_angle_between, rotate
from copy import deepcopy
import numpy as np
from typing import Optional


def kartview(env, x, make_relative_to_kart: bool = True, adapt_to_kart_rotation: bool = True):
    """Returns a vector in the kart frame (X right, Y up, Z forwards)

    Args:
        env (Env): The environment.
        x (np.ndarray): The vector to convert.
        make_relative_to_kart (bool): Whether to make the vector relative to the kart position.
        adapt_to_kart_rotation (bool): Whether to adapt the vector to the kart rotation.

    Returns:
        np.ndarray: The vector in the kart frame.
    """
    kart = env.unwrapped.world.karts[env.unwrapped.kart_ix]
    x_proj = x
    if make_relative_to_kart:
        x_proj = x - kart.location
    if adapt_to_kart_rotation:
        x_proj = rotate(x_proj, kart.rotation)
    return x_proj


class EnhanceObservations(Wrapper):
    """An observation wrapper that adds more informative observations."""

    def __init__(self, env: Env, previous_action_horizon: int = 1):
        """An observation wrapper that adds more informative observations.

        Args:
            env (Env): The environment to wrap.
            previous_action_horizon (int): The number of previous actions to consider.
        """
        super().__init__(env)

        self.observation_space = deepcopy(env.observation_space)
        # Update the observation space to include the new phase representation
        self.observation_space['phase'] = Discrete(
            n=len(Phase)
        )
        self.observation_space['nb_steps_since_set'] = Box(
            low=0,
            high=float('inf'),
            shape=(),
            dtype=np.int32,
        )
        self.observation_space['should_accelerate_in_start'] = Discrete(
            n=2
        )
        # Update with the new powerup state
        self.observation_space['powerup'] = Discrete(
            n=len(Powerup)
        )
        # Save number of steps to track phases
        self.nb_steps = 0
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
            n=2
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
        """Adds more informative observations.

        Args:
            observation (dict): The original observation.
            action (Optional[dict]): The action taken.
            did_nitro (bool): Whether nitro was used in this step.
            did_zipper (bool): Whether zipper was used in this step.
            reset (bool): Whether the environment is being reset.

        Returns:
            dict: The enhanced observation.
        """
        # Shalow copy
        observation = {**observation}
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
        return observation

    def reset(self, **kwargs):
        """Reset the environment and the step counter.

        Returns:
            observation (dict): The initial observation.
            info (dict): Additional information.
        """
        observation, info = self.env.reset(**kwargs)
        self.nb_steps = 0
        new_observation = self.get_new_observation(observation, action=None, did_nitro=False, did_zipper=False, reset=True)
        return new_observation, info

    def step(self, action):
        """Step the environment and update the step counter.

        Args:
            action: The action to take.

        Returns:
            observation (dict): The observation after taking the action.
            reward (float): The reward received after taking the action.
            terminated (bool): Whether the episode has terminated.
            truncated (bool): Whether the episode has been truncated.
            info (dict): Additional information.
        """
        kart = self.env.unwrapped.world.karts[self.env.unwrapped.kart_ix]
        did_nitro = action['nitro']
        did_zipper = (kart.powerup.type.value == Powerup.ZIPPER.value) and action['fire']
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.nb_steps += 1
        new_observation = self.get_new_observation(observation, action=action, did_nitro=did_nitro, did_zipper=did_zipper, reset=False)
        return new_observation, reward, terminated, truncated, info


class StuckObservations(Wrapper):
    """An observation wrapper that adds a 'stuck' indicator to the observation."""

    def __init__(self, env: Env):
        """An observation wrapper that adds a 'stuck' indicator to the observation.

        Args:
            env (Env): The environment to wrap.
        """
        super().__init__(env)

        # Update the observation space to include the 'stuck' indicator
        self.observation_space = deepcopy(env.observation_space)
        self.is_stuck = False
        self.is_being_rescued = False
        self.observation_space['stuck'] = Discrete(
            n=2,  # 0: not stuck, 1: stuck
        )
        self.time_since_stuck = 0.0
        self.observation_space['time_since_stuck'] = Box(
            low=0.0,
            high=5.0,
            shape=(),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        """Reset the environment and the stuck indicators.

        Returns:
            observation (dict): The initial observation.
            info (dict): Additional information.
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

        Args:
            action: The action to take.

        Returns:
            observation (dict): The observation after taking the action.
            reward (float): The reward received after taking the action.
            terminated (bool): Whether the episode has terminated.
            truncated (bool): Whether the episode has been truncated.
            info (dict): Additional information.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = self.get_new_observation(observation)
        return observation, reward, terminated, truncated, info

    def get_new_observation(self, observation):
        """Add 'stuck' and 'time_since_stuck' indicators to the observation.

        Args:
            observation (dict): The original observation.

        Returns:
            dict: The observation with 'stuck' and 'time_since_stuck' indicators added.
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


class ExpertObservations(Wrapper):
    """An observation wrapper that adds expert observations."""

    def __init__(self, env: Env, avoid_bad_items: bool = True, avoid_karts: bool = True):
        """An observation wrapper that adds expert observations.

        Args:
            env (Env): The environment to wrap.
            avoid_bad_items (bool): Whether to consider avoiding bad items.
            avoid_karts (bool): Whether to consider avoiding karts.
        """
        super().__init__(env)
        self.avoid_bad_items = avoid_bad_items
        self.avoid_karts = avoid_karts
        self.observation_space = deepcopy(env.observation_space)
        # Track direction
        self.observation_space['track_direction'] = Discrete(
            n=len(DirectionType)
        )
        # Speed to radius ratio
        self.observation_space['speed_radius_ratio'] = Box(
            low=0.0,
            high=float('inf'),
            shape=(),
            dtype=np.float32,
        )
        # Target position
        self.observation_space['target_position'] = Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(3,),
            dtype=np.float32,
        )
        # Curve duration
        self.observation_space['curve_duration'] = Box(
            low=0.0,
            high=5.0,
            shape=(),
            dtype=np.float32,
        )
        # Avoid obstacle
        self.observation_space['avoid_obstacle'] = Discrete(
            n=2,
        )
        # Obstacle position
        self.observation_space['obstacle_position'] = Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(3,),
            dtype=np.float32,
        )
        # Obstacle velocity
        self.observation_space['obstacle_velocity'] = Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(3,),
            dtype=np.float32,
        )

    def get_new_observation(self, observation):
        """Get a new observation with expert observations added.

        Args:
            observation (dict): The original observation.
        """
        # Shallow copy
        observation = {**observation}
        # Determine track direction
        direction_type, last_path_id = self.__determine_track_direction(observation)
        # Get track curve
        curve_center, curve_radius = self.__determine_turn_radius(observation, last_path_id)
        # Get kart speed
        kart_speed = np.linalg.norm(observation['velocity']).item()
        # Get speed ratio to curve radius (this will enable deciding whether to brake or not)
        speed_radius_ratio = kart_speed / curve_radius if curve_radius is not None else 0.0
        # Get target position
        if observation['stuck']:
            # If stuck, target the next path start
            target_position = observation['paths_start'][1]
        else:
            # Otherwise, get the target path ID
            target_path_id = self.__get_target_path_id(observation)
            # Get the target position
            target_position = (observation['paths_start'][target_path_id] + observation['paths_end'][target_path_id]) / 2.0
        # Get curve length
        if (last_path_id is not None) and (curve_radius is not None) and (curve_center is not None):
            last_path_center = (observation['paths_start'][last_path_id] + observation['paths_end'][last_path_id]) / 2.0
            diff_kart = -curve_center
            diff_last = last_path_center - curve_center
            angle = get_xz_angle_between(diff_kart, diff_last)
            curve_length = curve_radius * abs(angle)
        else:  # If the track direction is undefined
            curve_length = None
        # Duration to complete the curve at current speed and clamped to 5 seconds max
        curve_duration = min(curve_length / kart_speed if (curve_length is not None and kart_speed > 0) else 0.0, 5.0)
        # Get items ahead
        items_to_collect, items_to_avoid = self.__get_items_ahead(observation)
        # Get obstacles ahead
        avoid_obstacle, obstacle_position, observation_velocity = self.__get_obstacle_info(observation, target_position, items_to_collect, items_to_avoid)
        # Add to observation
        observation['track_direction'] = direction_type.value
        observation['speed_radius_ratio'] = speed_radius_ratio
        observation['target_position'] = target_position
        observation['curve_duration'] = curve_duration
        observation['avoid_obstacle'] = int(avoid_obstacle)
        observation['obstacle_position'] = obstacle_position if obstacle_position is not None else np.full((3,), 0.0)
        observation['obstacle_velocity'] = observation_velocity if observation_velocity is not None else np.full((3,), 0.0)
        # Return the new observation
        return observation

    def reset(self, **kwargs):
        """Reset the environment and add expert observations."""
        obs, info = self.env.reset(**kwargs)
        new_obs = self.get_new_observation(obs)
        return new_obs, info

    def step(self, action):
        """Step the environment and add expert observations."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_obs = self.get_new_observation(obs)
        return new_obs, reward, terminated, truncated, info

    def __determine_track_direction(self, observation):
        """Determines the direction of the track based on the upcoming path segments.

        Args:
            observation (dict): The current observation of the environment.

        Returns:
            DirectionType: The direction type of the track.
        """
        # Get the maximum number of path segments to consider
        max_nb_paths = len(observation['paths_start'])
        assert max_nb_paths >= 4, "There should be at least 4 path segments to determine the track direction"
        # Get the center point of the upcoming path segment
        next_path_center = (observation['paths_start'][1] + observation['paths_end'][1]) / 2.0
        # Get velocity vector
        velocity = observation['velocity']
        # Compute the angle between the velocity and the next path center
        velocity_to_next_angle = get_xz_angle_between(velocity, next_path_center)
        # If the angle is quite big, we consider the direction to be undefined to avoid
        # taking wrong decisions until we get back on track
        if abs(velocity_to_next_angle) > 0.3 * np.pi:
            return DirectionType.DIR_UNDEFINED, None
        # Determine if the track is going left, right or straight
        # Get the first angle (between direction to next path and its center direction to the path next to it)
        # TODO: debug why this isn't working as expected
        current_path_center = next_path_center
        next_path_center = (observation['paths_start'][2] + observation['paths_end'][2]) / 2.0
        v1 = current_path_center
        v2 = next_path_center - current_path_center
        angle1 = get_xz_angle_between(v1, v2)
        # Get second angle (between path center direction and the next path center direction)
        current_path_center = next_path_center
        next_path_center = (observation['paths_start'][3] + observation['paths_end'][3]) / 2.0
        v1 = v2
        v2 = next_path_center - current_path_center
        angle2 = get_xz_angle_between(v1, v2)
        rel_angle = angle1 + angle2
        # If the relative angle is small enough, consider it straight
        if abs(rel_angle) < ExpertAgent.MAX_STRAIGHT_ANGLE:
            rel_angle = 0.0
        # Iterate through the next path segments to determine the overall direction
        last_path_id = 3
        for i in range(3, max_nb_paths - 1):
            current_path_center = next_path_center
            next_path_center = (observation['paths_start'][i + 1] + observation['paths_end'][i + 1]) / 2.0
            v1 = v2
            v2 = next_path_center - current_path_center
            angle2 = get_xz_angle_between(v1, v2)
            new_rel_angle = angle1 + angle2
            # If the relative angle is small enough, consider it straight
            if abs(new_rel_angle) < ExpertAgent.MAX_STRAIGHT_ANGLE:
                new_rel_angle = 0.0
            # If the new relative angle contradicts the previous one, stop here as it means the track changed direction
            if (rel_angle != 0 and new_rel_angle == 0) or (rel_angle == 0 and new_rel_angle != 0) or (rel_angle * new_rel_angle < 0):
                break
            # Save angles for next iteration
            angle1 = angle2
            # Otherwise, update the relative angle
            rel_angle = new_rel_angle
            last_path_id = i + 1
        # Determine the direction type based on the relative angle
        assert rel_angle is not None, "rel_angle should have been set by now"
        direction_type = DirectionType.DIR_STRAIGHT if rel_angle == 0.0 else (DirectionType.DIR_RIGHT if rel_angle > 0 else DirectionType.DIR_LEFT)
        # Return the direction type and the last path segment id considered
        return direction_type, last_path_id

    def __determine_turn_radius(self, observation, last_path_id: Optional[int]):
        """Determine the center and radius of the turn based on the last path segment.

        Args:
            observation (dict): The current observation from the environment.
            last_path_id (Optional[int]): The index of the last path segment considered.

        Returns:
            tuple[Optional[np.ndarray], Optional[float]]: The center of the turn (as a 3D point) and the radius of the turn.
        """
        # If last_path_id is None, we cannot determine the turn radius
        if last_path_id is None:
            return None, None
        # Get the center point of the last path segment
        last_path_center = (observation['paths_start'][last_path_id] + observation['paths_end'][last_path_id]) / 2.0
        # Compute the mid point and orthogonal vector
        mid = 0.5 * last_path_center
        direction = last_path_center
        orthogonal = np.array([direction[2], 0.0, -direction[0]])
        q1 = mid + orthogonal
        # Compute line equation between mid and q1
        line1_a = (q1[2] - mid[2]) / (q1[0] - mid[0]) if (q1[0] - mid[0]) != 0 else None
        line1_b = mid[2] - line1_a * mid[0] if line1_a is not None else None
        # Compute line intersection with x axis
        if line1_a is not None:
            # If there is an intersection, calculate its coordinates
            intersect_x = (-line1_b) / line1_a
            intersect_z = line1_a * intersect_x + line1_b
            # Set the curve center to be the intersection point and radius to be its xz norm
            curve_center = np.array([intersect_x, mid[1], intersect_z])
            curve_radius = np.linalg.norm(curve_center).item()
        else:
            # If there is no intersection, assume that the curve is a semi circle around the mid point
            curve_center = mid
            curve_radius = 0.5 * np.linalg.norm(last_path_center).item()
        # Return the curve center and radius
        return curve_center, curve_radius

    def __get_target_path_id(self, observation):
        """Returns the target path ID based on the current state.

        Args:
            observation (dict): The current observation from the environment.

        Returns:
            int: The target path ID.
        """
        # By default, we aim at the next path segment
        # as it's the safest choice (kart won't go off track)
        target_path_id = 1

        # Aiming at paths further ahead is better in most cases
        # as it allows to better anticipate turns but it's more risky
        # To be able to aim further, we need to make sure that the kart can
        # stay within the track boundaries until that path segment

        # We do that by checking the angle range defined by the end edges of the path segments
        # We aim at the furthest path segment whose center is still within an angle range of at least 45 degrees

        # Get current path direction
        current_path_dir = observation['paths_end'][0] - observation['paths_start'][0]
        # Get current path end edges
        left_edge, right_edge = self.__get_path_end_edges(observation, 0)
        # Compute angles between path direction and edges
        angle_left = get_xz_angle_between(current_path_dir, left_edge)
        angle_right = get_xz_angle_between(current_path_dir, right_edge)
        # Initialize target angle range
        target_angle_min = angle_left
        target_angle_max = angle_right
        # Get max number of paths
        max_nb_paths = len(observation['paths_start'])
        # Iterate over future path segments to find the furthest safe target
        for path_id in range(1, max_nb_paths - 1):
            # Get next path end edges
            left_edge, right_edge = self.__get_path_end_edges(observation, path_id)
            # Compute angles between next path direction and edges
            angle_left = get_xz_angle_between(current_path_dir, left_edge)
            angle_right = get_xz_angle_between(current_path_dir, right_edge)
            # Update target angle range
            target_angle_min = max(target_angle_min, angle_left)
            target_angle_max = min(target_angle_max, angle_right)
            # Get center of the next path segment
            next_path_center = (observation['paths_start'][path_id + 1] + observation['paths_end'][path_id + 1]) / 2.0
            # If the center of the next path segment is outside the target angle range, or the angle range got too narrow (less than 45 degrees), stop here
            angle_to_next_path = get_xz_angle_between(current_path_dir, next_path_center)
            if (target_angle_max - target_angle_min < 0.785) or (angle_to_next_path <= target_angle_min) or (angle_to_next_path >= target_angle_max):
                return target_path_id
            # Otherwise, update the target path ID
            target_path_id = path_id + 1
        # Return the target path ID
        return target_path_id

    def __get_path_end_edges(self, observation, path_id: int):
        """Returns the left and right edges of the end of a given path segment.

        Args:
            observation (dict): The current observation from the environment.
            path_id (int): The ID of the path segment.

        Returns:
            tuple[np.ndarray, np.ndarray]: The left and right edges of the end of the path segment.
        """
        path_start = observation['paths_start'][path_id]
        path_end = observation['paths_end'][path_id]
        path_width = observation['paths_width'][path_id][0]
        path_dir = path_end - path_start
        side_vec = np.array([path_dir[2], 0.0, -path_dir[0]])  # to the right
        side_vec /= np.linalg.norm(side_vec)  # normalize
        left_edge = path_end - (path_width / 2.0) * side_vec
        right_edge = path_end + (path_width / 2.0) * side_vec
        return left_edge, right_edge

    def __get_items_ahead(self, observation):
        """Returns a list of item IDs that are ahead of the kart in the current path segment.

        Args:
            observation (dict): The current observation from the environment.
        Returns:
            tuple[list[tuple[int, int]], list[tuple[int, int]]]: Lists of item IDs to collect and avoid along with their associated path IDs.
        """
        current_path_id = 0
        items_to_collect = []
        items_to_avoid = []
        # 1) Filter and sort all items close by
        state_paths = len(observation['paths_start'])
        while current_path_id < state_paths:
            items_ahead = [i for (i, pos) in enumerate(observation['items_position']) if self.__check_if_pos_in_path(observation, pos, current_path_id)]
            for item_id in items_ahead:
                # Skip items that are behind the kart when checking the first path segment
                if current_path_id == 0:
                    # Get item position
                    pos = observation['items_position'][item_id]
                    # Get path start and end
                    path_start = observation['paths_start'][current_path_id]
                    path_end = observation['paths_end'][current_path_id]
                    path_dir = path_end - path_start
                    # Get path length and normalize direction
                    path_length = np.linalg.norm(path_dir)
                    path_dir /= path_length
                    # Compute relative position of item and kart to path start
                    rel_pos = pos - path_start
                    kart_rel_pos = -path_start
                    # Project relative positions onto path direction
                    proj_length = np.dot(rel_pos, path_dir)
                    kart_proj_length = np.dot(kart_rel_pos, path_dir)
                    # If item is behind the kart, skip it
                    if proj_length <= kart_proj_length:
                        continue
                # Evaluate item for collection or avoidance
                self.__evaluate_items(observation, item_id, current_path_id, items_to_collect, items_to_avoid)
            current_path_id += 1
        # Return the lists of items to collect and avoid
        return items_to_collect, items_to_avoid

    def __get_obstacle_info(self, observation, target_point: np.ndarray, items_to_collect: list[tuple[int, int]], items_to_avoid: list[tuple[int, int]]):
        """Analyzes obstacles (items to avoid and karts) and computes a new target point if avoidance is necessary.

        Args:
            observation (dict): The current observation from the environment.
            target_point (np.ndarray): The target point to steer towards.
            items_to_collect (list[tuple[int, int]]): List to store item IDs to collect and their associated path IDs.
            items_to_avoid (list[tuple[int, int]]): List to store item IDs to avoid and their associated path IDs.
        """
        # Get current speed
        speed = np.linalg.norm(observation['velocity'])
        # 1) If there are items to avoid, and we are going towards them, avoid them
        if self.avoid_bad_items and (len(items_to_avoid) > 0) and (speed > 0):
            for (item_id, path_id) in items_to_avoid:
                item_to_avoid_position = observation['items_position'][item_id]
                angle = get_xz_angle_between(np.array([0.0, 0.0, 1.0]), item_to_avoid_position)
                if abs(angle) < 0.3:
                    dist_to_item = np.linalg.norm(item_to_avoid_position).item()
                    nb_max_steps = 4
                    nb_steps_to_reach = ((dist_to_item - ExpertAgent.DISTANCE_ASIDE_TO_AVOID_ITEM - ExpertAgent.KART_LENGTH / 2.0) / speed) / self.env.unwrapped.config.step_size
                    if nb_steps_to_reach < nb_max_steps:
                        # Return that we need to avoid this item and the new target point
                        return True, item_to_avoid_position, np.array([0.0, 0.0, 0.0])
        # 2) If there are karts that may collide with us, avoid them
        # But if our kart has a bomb attached to it, we won't try to avoid other karts
        # as we may want to hit them on purpose to pass the bomb to them
        if self.avoid_karts and (observation['attachment'] != Attachment.BOMB.value):
            for i, kart_position in enumerate(observation['karts_position']):
                if observation['karts_rel_rank'][i] < 0:
                    continue  # Only consider karts ahead
                kart_speed = np.linalg.norm(observation['karts_velocity'][i])
                relative_speed = speed - kart_speed
                if relative_speed <= 0:
                    continue  # Only consider karts that are slower than us
                angle = get_xz_angle_between(np.array([0.0, 0.0, 1.0]), kart_position)
                if abs(angle) < 0.3:
                    dist_to_kart = np.linalg.norm(kart_position).item()
                    nb_max_steps = 4
                    nb_steps_to_reach = ((dist_to_kart - ExpertAgent.KART_LENGTH) / relative_speed) / self.env.unwrapped.config.step_size
                    if nb_steps_to_reach < nb_max_steps:
                        # Return that we need to avoid this kart and the new target point
                        return True, kart_position, observation['karts_velocity'][i]
        # Else we don't need to avoid anything
        return False, None, None

    def __check_if_pos_in_path(self, observation, position: np.ndarray, path_id: int):
        """Checks if a given position is within the path segment.

        Args:
            observation (dict): The current observation from the environment.
            position (np.ndarray): The position to check.
            path_id (int): The ID of the path segment.

        Returns:
            bool: True if the position is within the path segment, False otherwise.
        """
        path_start = observation['paths_start'][path_id]
        path_end = observation['paths_end'][path_id]
        path_dir = path_end - path_start
        path_length = np.linalg.norm(path_dir)
        if path_length == 0:
            return False
        path_dir /= path_length
        rel_pos = position - path_start
        proj_length = np.dot(rel_pos, path_dir)
        proj_width = np.linalg.norm(rel_pos - proj_length * path_dir)
        half_path_width = 0.5 * observation['paths_width'][path_id][0]
        return (0 <= proj_length) and (proj_length <= path_length) and (proj_width <= half_path_width - 0.5 * ExpertAgent.KART_WIDTH)

    def __evaluate_items(self, observation, item_id: int, current_path_id: int, items_to_collect: list[tuple[int, int]], items_to_avoid: list[tuple[int, int]]):
        """Evaluates items to collect and avoid based on the kart's aim direction.

        Args:
            observation (dict): The current observation from the environment.
            item_id (int): The ID of the item to evaluate.
            current_path_id (int): The ID of the current path segment.
            items_to_collect (list[tuple[int, int]]): List to store item IDs to collect and their associated path IDs.
            items_to_avoid (list[tuple[int, int]]): List to store item IDs to avoid and their associated path IDs.
        """
        item_position = observation['items_position'][item_id]
        item_type = observation['items_type'][item_id]

        avoid = False
        # Negative items to avoid
        if item_type == ItemType.BUBBLEGUM.value:
            avoid = True
        elif item_type == ItemType.BANANA.value:
            avoid = True
        # Positive items to collect
        elif item_type == ItemType.NITRO_BIG.value:
            # Only collect nitro, if it can actually be stored.
            if observation['energy'] == ExpertAgent.NITRO_MAX:
                return
        elif item_type == ItemType.NITRO_SMALL.value:
            # Only collect nitro, if it can actually be stored.
            if observation['energy'] == ExpertAgent.NITRO_MAX:
                return

        # Now insert the item into the sorted list of items to avoid
        # (or to collect). The lists are (for now) sorted by distance
        if avoid:
            target_list = items_to_avoid
        else:
            target_list = items_to_collect

        insert_index = 0
        item_distance = np.linalg.norm(item_position).item()
        for idx, _ in target_list:
            other_item_position = observation['items_position'][idx]
            other_item_distance = np.linalg.norm(other_item_position).item()
            if item_distance < other_item_distance:
                break
            insert_index += 1

        target_list.insert(insert_index, (item_id, current_path_id))


class CustomPolarObservations(ObservationWrapper):
    """Modifies position to polar positions

    Angles are in radian

    input: X right, Y up, Z forwards
    output: (angle in the ZX plane, angle in the ZY plane, distance)
    """
    #: Keys to transform (batch)
    KEYS = ["items_position", "karts_position", "paths_start", "paths_end", "karts_velocity"]
    #: Keys to transform (single)
    SIMPLE_KEYS = ["target_position", "obstacle_position", "center_path", "velocity", "obstacle_velocity"]

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
