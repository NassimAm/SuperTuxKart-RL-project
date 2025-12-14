import math
from core.state import State, Phase, Attachment, Powerup, ItemType, DirectionType
from core.action import Action
from core.utils import from_polar_to_cartesian, get_xz_angle_between
from typing import Optional
import numpy as np
from gymnasium import Env
from pystk2_gymnasium.envs import STKRaceEnv


class ExpertAgent:
    """An expert agent for SuperTuxKart that uses predefined strategies to navigate the track. This implementation is a
    simplified implementation inspired from the original C++ expert agent from SuperTuxKart. The original code that we are
    referring to can be found here https://github.com/bpiwowar/pystk2/blob/master/src/karts/controller/skidding_ai.cpp"""

    env: STKRaceEnv
    """The environment in which the agent operates."""
    enable_drift: bool
    """Whether to enable drifting behavior."""
    avoid_bad_items: bool
    """Whether to avoid bad items on the track."""
    avoid_karts: bool
    """Whether to avoid karts on the track."""
    use_nitro: bool
    """Whether to use nitro boosts."""
    use_items: bool
    """Whether to use items on the track."""
    collect_items: bool
    """Whether to collect items on the track."""
    max_speed: float
    """The maximum speed of the kart based on difficulty level."""
    path_decision_angle: int = 45  # degrees
    """The angle threshold for path decision-making."""

    # ============================== Path characteristics ==============================
    MAX_STRAIGHT_ANGLE = 0.1  # radians
    # ============================== Kart characteristics ==============================
    # NOTE: These values assume that we always have the same type of kart (the default penguin kart)
    # see https://github.com/bpiwowar/pystk2/blob/f86f21efbf137c95c26f85d113dd1525a8fc4ea6/data/kart_characteristics.xml for more details
    # TURN_RADIUS = {0: 2.3, 10: 8.625, 25: 17.25, 45: 34.5}
    # nitro
    NITRO_DURATION = 1.0  # seconds
    NITRO_FADE_OUT_TIME = 2.4  # seconds
    NITRO_MAX_SPEED_INCREASE = 5.0  # m/s
    NITRO_CONSUMPTION = 1.25
    # NITRO_SMALL_CONTAINER = 1.0
    # NITRO_BIG_CONTAINER = 3.0
    NITRO_MAX = 20.0
    # zipper
    ZIPPER_DURATION = 3.5  # seconds
    ZIPPER_FADE_OUT_TIME = 1.0  # seconds
    ZIPPER_MAX_SPEED_INCREASE = 15.0  # m/s
    # skidding
    # SKID_BONUS_SPEED = [4.5, 6.5]  # m/s
    SKID_TIME_TILL_BONUS = [1.0, 3.0]
    SKID_REDUCE_TURN_MIN = 0.2
    SKID_REDUCE_TURN_MAX = 0.8
    SKID_MIN_SPEED = 10  # m/s
    # parachute
    PARACHUTE_DURATION_OTHER = 2.5  # seconds
    # swatter
    SWATTER_RADIUS = 3.0  # meters
    # general characteristics
    MAX_SPEED = 25.0  # m/s
    MAX_SPEED_FACTORS = {
        0: 0.6,
        1: 0.8,
        2: 0.92,
        3: 1.0
    }
    WHEEL_BASE = 1.0084210634231567  # meters
    KART_WIDTH = 0.821  # meters
    KART_HEIGHT = 0.675  # meters
    KART_LENGTH = 1.437  # meters
    # ============================= Configuration Assumptions ==============================
    # NOTE: These values are assumptions made about the environment configuration
    # They are drown from here https://github.com/bpiwowar/pystk2/blob/f86f21efbf137c95c26f85d113dd1525a8fc4ea6/data/stk_config.xml
    # SHIELD_RESTRICT_WEAPONS = False
    # ============================== Personal Assumptions ==============================
    # NOTE: These values are personal assumptions made to simplify the implementation
    DISTANCE_ASIDE_TO_AVOID_ITEM = 2.0  # meters
    # braking
    MAX_SPEED_RADIUS_RATIO = 1.3
    BRAKE_MIN_SPEED = 5.0  # m/s
    # firing
    MAX_TIME_SINCE_LAST_FIRE = 5.0  # seconds

    def __init__(self, env: Env, enable_drift: bool = True,
                 avoid_bad_items: bool = True, avoid_karts: bool = True,
                 use_nitro: bool = True, use_items: bool = True, collect_items: bool = False,
                 path_decision_angle: int = 45):
        """Initializes the ExpertAgent.

        Args:
            enable_drift (bool): Whether to enable drifting behavior. Default is True.
            avoid_bad_items (bool): Whether to avoid bad items on the track. Default is True.
            avoid_karts (bool): Whether to avoid karts on the track. Default is True.
            use_nitro (bool): Whether to use nitro boosts. Default is True.
            use_items (bool): Whether to use items on the track. Default is True.
        """
        self.env = env.unwrapped
        self.max_speed = ExpertAgent.MAX_SPEED * ExpertAgent.MAX_SPEED_FACTORS[self.env.difficulty]
        self.enable_drift = enable_drift
        self.avoid_bad_items = avoid_bad_items
        self.avoid_karts = avoid_karts
        self.use_nitro = use_nitro
        self.use_items = use_items
        self.collect_items = collect_items
        self.path_decision_angle = path_decision_angle

    def act(self, state: dict):
        """Determines the action to take based on the current state.

        Args:
            state (dict): The current state of the environment.

        Returns:
            dict: The action to take in the expected environment format.
        """
        # Convert the state dictionary to a State object
        state: State = State.from_env(state)
        # Initialize action components
        action = Action.empty()
        # If the kart needs to be rescued, do it now (and nothing else)
        if state['stuck']:
            self.__handle_rescue(state, action)
            return Action.to_env(action)
        # If the race didn't start yet, handle that special case
        if state['phase'].value < Phase.RACE_PHASE.value:
            self.__handle_race_start(state, action)
            return Action.to_env(action)
        # Get the target path ID
        target_path_id = self.__get_target_path_id(state)
        # Get the target position
        target_position = (from_polar_to_cartesian(state['paths_start'][target_path_id]) + from_polar_to_cartesian(state['paths_end'][target_path_id])) / 2.0
        # Determine the track direction
        track_direction, last_path_id = self.__determine_track_direction(state, target_position)
        # Get track curve
        curve_center, curve_radius = self.__determine_turn_radius(state, last_path_id)
        # Handle acceleration and braking
        self.__handle_acceleration_and_braking(state, action, track_direction, last_path_id, curve_radius)
        # Handle steering
        self.__handle_steering(state, action, target_position, track_direction, last_path_id, curve_center, curve_radius)
        # Return the action in the expected environment format
        return Action.to_env(action)

    def __handle_race_start(self, state: State, action: Action):
        """Handles the race start logic, including false starts and start delays.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
        """
        # If the agent didn't reach the delay yet, do nothing
        if not state['should_accelerate_in_start']:
            return
        # otherwise if delay passed, accelerate to the max
        action['acceleration'] = 1.0

    def __handle_acceleration_and_braking(self, state: State, action: Action, track_direction: DirectionType,
                                          last_path_id: Optional[int], curve_radius: Optional[float]):
        """Handles acceleration and braking logic based on the current state.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
            track_direction (DirectionType): The direction type of the track.
            last_path_id (Optional[int]): The ID of the last path segment.
            curve_radius (Optional[float]): The radius of the turn.
        """
        # Get max turn speed
        max_turn_speed = float('inf') if curve_radius is None else ExpertAgent.MAX_SPEED_RADIUS_RATIO * curve_radius
        # Handle braking
        self.__handle_braking(state, action, track_direction, max_turn_speed)
        # If we are braking, set acceleration to 0
        if action['brake']:
            action['acceleration'] = 0.0
            return
        # Handle nitro
        if self.use_nitro:
            self.__handle_nitro(state, action, max_turn_speed)
        # Handle zipper
        if self.use_items:
            self.__handle_zipper(state, action, track_direction, last_path_id)
        # Otherwise, set acceleration to max
        action['acceleration'] = 1.0

    def __handle_braking(self, state: State, action: Action, track_direction: DirectionType, max_turn_speed: float):
        """Handles braking logic based on the current state.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
            track_direction (DirectionType): The direction type of the track.
            max_turn_speed (float): The maximum speed allowed for the turn.
        """
        # Set default to no braking
        action['brake'] = False
        # Get current speed
        speed = state['velocity'][2]
        # If the current track direction is right or left, speed is above max turn speed and kart is steering too much, brake
        if (track_direction == DirectionType.DIR_RIGHT) or (track_direction == DirectionType.DIR_LEFT):
            steer_value = state['prev_actions'][-1]['steer']
            if (speed > ExpertAgent.BRAKE_MIN_SPEED) and (speed > max_turn_speed) and (abs(steer_value) > 0.95):
                action['brake'] = True
            return

    def __handle_nitro(self, state: State, action: Action, max_safe_speed: float):
        """Handles nitro logic based on the current state.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
            max_safe_speed (float): The maximum safe speed to use nitro or zipper.
        """
        # Set nitro to false by default
        action['nitro'] = False
        # Get time until fadeout
        time_until_fadeout = state['nitro_duration_time_left'] + state['nitro_fadeout_time_left']
        # Set energy reserve
        energy_reserve = 12
        # Don't use nitro if we are braking
        if action['brake']:
            return
        # Don't use nitro if the kart is not on ground
        if state['jumping']:
            return
        speed = state['velocity'][2]
        if speed + ExpertAgent.NITRO_MAX_SPEED_INCREASE > max_safe_speed:
            return
        # If a parachute or anvil is attached, the nitro don't give much
        # benefit. Better wait till later.
        has_slowdown_attachment = (state['attachment'] == Attachment.PARACHUTE) or (state['attachment'] == Attachment.ANVIL)
        if has_slowdown_attachment:
            return
        # Don't compute nitro usage if we don't have nitro
        if state['energy'] == 0:
            return
        # If the AI has a lot of nitro, it should consume it without waiting for some fadeout
        big_reserve = False
        # If this kart is the last kart, set nitro reserve to be at most 2
        if state['rank'] == 0.0:
            energy_reserve = 2
        # Estimate time towards the end of the race.
        # Decreases the reserve size when there is an estimate of time remaining
        # to the end of less than 2,5 times the maximum nitro effect duration.
        # This vary depending on kart characteristic.
        # There is a margin because the kart will go a bit faster than predicted
        # by the estimate, because the kart may collect more nitro and because
        # there may be moments when it's not useful to use nitro (parachutes, etc).
        if energy_reserve > 0:
            finish = self.__estimate_finish_time(state) - state['game_time']
            max_time_effect = (ExpertAgent.NITRO_DURATION + ExpertAgent.NITRO_FADE_OUT_TIME) / ExpertAgent.NITRO_CONSUMPTION * state['energy'] * 2
            if time_until_fadeout > 0:
                energy_reserve = 0
            if (2.5 * max_time_effect) >= finish:
                # Absolute reduction to avoid small amount of unburned nitro at the end
                energy_reserve = min(energy_reserve, finish / (2.5 * max_time_effect / state['energy']) - 0.5)
            if ((1.8 * max_time_effect) >= finish) or (state['energy'] >= 16):
                big_reserve = True
        # Don't use nitro if there is already a nitro boost active
        # Nitro effect and fadeout may vary between karts type
        # So vary time according to kart properties
        if (time_until_fadeout >= (ExpertAgent.NITRO_FADE_OUT_TIME * 0.5)) and (not big_reserve):
            return
        # If trying to pass a bomb to a kart faster or far ahead, use nitro reserve
        speed = state['velocity'][2]
        if (state['attachment'] == Attachment.BOMB) and (energy_reserve > 0):
            karts_rel_rank = state['karts_rel_rank']
            kart_ahead_id = next((i for i in range(len(karts_rel_rank)) if karts_rel_rank[i] > 0), None)
            if (kart_ahead_id is not None) and ((state['karts_distance'][kart_ahead_id] > 10.0) or (state['karts_velocity'][kart_ahead_id][2] > speed)):
                energy_reserve = 0
        # Don't use nitro if building an energy reserve
        if state['energy'] <= energy_reserve:
            return
        if (speed < 5 and speed > 2) or (speed > 0.96 * self.max_speed) or (speed > 0.3 * self.max_speed and big_reserve):
            action['nitro'] = True
            return

    def __handle_zipper(self, state: State, action: Action, track_direction: DirectionType, last_path_id: Optional[int]):
        """Handles zipper logic based on the current state.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
            track_direction (DirectionType): The direction type of the track.
            last_path_id (Optional[int]): The ID of the last path segment.
        """
        # Don't use zipper if we are braking
        if action['brake']:
            return
        # Don't use zipper if the kart is not on ground
        if state['jumping']:
            return
        # If a parachute or anvil is attached, the zipper don't give much
        # benefit. Better wait till later.
        has_slowdown_attachment = (state['attachment'] == Attachment.PARACHUTE) or (state['attachment'] == Attachment.ANVIL)
        if has_slowdown_attachment:
            return
        # Get speed of the kart
        speed = state['velocity'][2]
        # If we have a zipper, are going fast enough and don't have an active zipper, consider using it
        if (state['powerup'] == Powerup.ZIPPER) and (speed > 1.0) and (state['zipper_duration_time_left'] == 0):
            # If the track is straight enough for long enough, use the zipper
            if track_direction == DirectionType.DIR_STRAIGHT:
                assert last_path_id is not None, "last_path_id should not be None here"
                last_path_center = (from_polar_to_cartesian(state['paths_start'][last_path_id]) + from_polar_to_cartesian(state['paths_end'][last_path_id])) / 2.0
                diff = np.linalg.norm(last_path_center).item()
                if diff > (speed + ExpertAgent.ZIPPER_MAX_SPEED_INCREASE) * self.env.config.step_size * 2.5:
                    action['fire'] = True
                    return

    def __handle_steering(self, state: State, action: Action, target_position: np.ndarray, track_direction: DirectionType, last_path_id: Optional[int],
                          curve_center: Optional[np.ndarray], curve_radius: Optional[float]):
        """Handles steering logic based on the current state.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
            target_position (np.ndarray): The target position to aim for.
            track_direction (DirectionType): The direction type of the track.
            last_path_id (Optional[int]): The index of the last path segment considered.
            curve_center (Optional[np.ndarray]): The center of the turn (as a 3D point).
            curve_radius (Optional[float]): The radius of the turn.
        """
        # Handle item collection and avoidance
        items_to_collect, items_to_avoid = self.__get_items_ahead(state)
        target_position = self.__handle_collection_and_avoidance(state, target_position, items_to_collect, items_to_avoid)
        # Handle item usage
        if self.use_items:
            self.__handle_items(state, action, items_to_collect, items_to_avoid)
        # Compute the required steer to reach the target point
        steer_angle = self.__steer_to_point(target_position)
        # Set the steering action
        self.__set_steering(state, action, steer_angle, track_direction, last_path_id, curve_center, curve_radius)

    def __handle_rescue(self, state: State, action: Action):
        """Handles rescue logic based on the current state.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
        """
        # Set acceleration to 0
        action['acceleration'] = 0.0
        # Set brake to True to go backwards
        action['brake'] = True
        # Set rescue to True
        action['rescue'] = True
        # keep going backwards if the time since stuck is less than 1 second
        if state['time_since_stuck'] < 1.0:
            return
        # otherwise, go straight forwards aiming for the next path segment
        action['brake'] = False
        action['acceleration'] = 1.0
        target_position = from_polar_to_cartesian(state['paths_start'][1])
        steer_angle = self.__steer_to_point(target_position)
        self.__set_steering(state, action, steer_angle, DirectionType.DIR_UNDEFINED, None, None, None)

    def __get_target_path_id(self, state: State):
        """Returns the target path ID based on the current state.

        Args:
            state (State): The current state of the environment.

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
        current_path_dir = from_polar_to_cartesian(state['paths_end'][0]) - from_polar_to_cartesian(state['paths_start'][0])
        # Get current path end edges
        left_edge, right_edge = self.__get_path_end_edges(state, 0)
        # Compute angles between path direction and edges
        angle_left = get_xz_angle_between(current_path_dir, left_edge)
        angle_right = get_xz_angle_between(current_path_dir, right_edge)
        # Initialize target angle range
        target_angle_min = angle_left
        target_angle_max = angle_right
        # Get max number of paths
        max_nb_paths = len(state['paths_start'])
        # Iterate over future path segments to find the furthest safe target
        for path_id in range(1, max_nb_paths - 1):
            # Get next path end edges
            left_edge, right_edge = self.__get_path_end_edges(state, path_id)
            # Compute angles between next path direction and edges
            angle_left = get_xz_angle_between(current_path_dir, left_edge)
            angle_right = get_xz_angle_between(current_path_dir, right_edge)
            # Update target angle range
            target_angle_min = max(target_angle_min, angle_left)
            target_angle_max = min(target_angle_max, angle_right)
            # Get center of the next path segment
            next_path_center = (from_polar_to_cartesian(state['paths_start'][path_id + 1]) + from_polar_to_cartesian(state['paths_end'][path_id + 1])) / 2.0
            # If the center of the next path segment is outside the target angle range, or the angle range got too narrow (less than 45 degrees), stop here
            angle_to_next_path = get_xz_angle_between(current_path_dir, next_path_center)
            if (target_angle_max - target_angle_min < math.radians(self.path_decision_angle)) or (angle_to_next_path <= target_angle_min) or (angle_to_next_path >= target_angle_max):
                return target_path_id
            # Otherwise, update the target path ID
            target_path_id = path_id + 1
        # Return the target path ID
        return target_path_id

    def __get_path_end_edges(self, state: State, path_id: int):
        """Returns the left and right edges of the end of a given path segment.

        Args:
            state (State): The current state of the environment.
            path_id (int): The ID of the path segment.

        Returns:
            tuple[np.ndarray, np.ndarray]: The left and right edges of the end of the path segment.
        """
        path_start = from_polar_to_cartesian(state['paths_start'][path_id])
        path_end = from_polar_to_cartesian(state['paths_end'][path_id])
        path_width = state['paths_width'][path_id]
        path_dir = path_end - path_start
        side_vec = np.array([path_dir[2], 0.0, -path_dir[0]])  # to the right
        side_vec /= np.linalg.norm(side_vec)  # normalize
        left_edge = path_end - (path_width / 2.0) * side_vec
        right_edge = path_end + (path_width / 2.0) * side_vec
        return left_edge, right_edge

    def __get_items_ahead(self, state: State):
        """Returns a list of item IDs that are ahead of the kart in the current path segment.

        Args:
            state (State): The current state of the environment.

        Returns:
            tuple[list[tuple[int, int]], list[tuple[int, int]]]: Lists of item IDs to collect and avoid along with their associated path IDs.
        """
        current_path_id = 0
        items_to_collect = []
        items_to_avoid = []
        # 1) Filter and sort all items close by
        state_paths = len(state['paths_start'])
        while current_path_id < state_paths:
            items_ahead = [i for (i, pos) in enumerate(state['items_position']) if self.__check_if_pos_in_path(state, pos, current_path_id)]
            for item_id in items_ahead:
                # Skip items that are behind the kart when checking the first path segment
                if current_path_id == 0:
                    # Get item position
                    pos = from_polar_to_cartesian(state['items_position'][item_id])
                    # Get path start and end
                    path_start = from_polar_to_cartesian(state['paths_start'][current_path_id])
                    path_end = from_polar_to_cartesian(state['paths_end'][current_path_id])
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
                self.__evaluate_items(state, item_id, current_path_id, items_to_collect, items_to_avoid)
            current_path_id += 1
        # Return the lists of items to collect and avoid
        return items_to_collect, items_to_avoid

    def __handle_items(self, state: State, action: Action, items_to_collect: list[tuple[int, int]], items_to_avoid: list[tuple[int, int]]):
        """Handles item collection and avoidance logic based on the current state.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
            items_to_collect (list[tuple[int, int]]): List to store item IDs to collect and their associated path IDs.
            items_to_avoid (list[tuple[int, int]]): List to store item IDs to avoid and their associated path IDs.
        """
        # If we don't have a powerup, nothing to do
        if state['powerup'] == Powerup.NOTHING:
            return
        # Get kart ahead ID
        karts_rel_rank = state['karts_rel_rank']
        kart_ahead_id = next((i for i in range(len(karts_rel_rank)) if karts_rel_rank[i] > 0), None)
        # Treat each powerup type accordingly
        if state['powerup'] == Powerup.BUBBLEGUM:
            self.__handle_bubblegum(state, action, items_to_collect, items_to_avoid)
        elif state['powerup'] == Powerup.CAKE:
            self.__handle_cake(state, action)
        elif state['powerup'] == Powerup.BOWLING:
            self.__handle_bowling(state, action)
        elif state['powerup'] == Powerup.PLUNGER:
            kart_ahead_distance = state['karts_distance'][kart_ahead_id] if kart_ahead_id is not None else float('inf')
            if kart_ahead_distance < 30.0:
                action['fire'] = True
        elif state['powerup'] == Powerup.SWITCH:
            self.__handle_switch(state, action, items_to_avoid)
        elif state['powerup'] == Powerup.PARACHUTE:
            # Wait until a previous parachute effect is gone
            if state['time_since_last_fire'] > ExpertAgent.PARACHUTE_DURATION_OTHER:
                action['fire'] = True
        elif state['powerup'] == Powerup.SWATTER:
            # if the kart has a shield, do not break it by using a swatter
            if state['attachment'] != Attachment.BUBBLEGUM_SHIELD:
                self.__handle_swatter(state, action, items_to_collect)
        elif state['powerup'] == Powerup.RUBBERBALL:
            # Perhaps some more sophisticated algorithm might be useful.
            # For now: fire if there is a kart ahead (which means that
            # this kart is certainly not the first kart)
            if state['rank'] != 1.0:
                action['fire'] = True

    def __handle_bubblegum(self, state: State, action: Action, items_to_collect: list[tuple[int, int]],
                           items_to_avoid: list[tuple[int, int]]):
        """Handles bubblegum powerup logic based on the current state.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
            items_to_collect (list[tuple[int, int]]): List to store item IDs to collect and their associated path IDs.
            items_to_avoid (list[tuple[int, int]]): List to store item IDs to avoid and their associated path IDs.
        """
        attachment = state['attachment']
        # Use shield to remove bad attachments
        if (attachment == Attachment.PARACHUTE) or (attachment == Attachment.ANVIL):
            action['fire'] = True
            return
        # if it is a bomb, wait : we may pass it to another kart before the timer runs out
        if attachment == Attachment.BOMB:
            if state['attachment_time_left'] < 2.0:
                action['fire'] = True
                return
        # Use shield if kart is going to hit a bad item (banana or bubblegum)
        speed = state['velocity'][2]
        if (state['attachment'] != Attachment.BUBBLEGUM_SHIELD) and (len(items_to_avoid) > 0):
            d = np.linalg.norm(from_polar_to_cartesian(state['items_position'][items_to_avoid[0][0]]))
            if (speed > 0) and (((d - ExpertAgent.DISTANCE_ASIDE_TO_AVOID_ITEM) / speed) < self.env.config.step_size * 2.5):
                action['fire'] = True
                return
        # Use shield if kart is going to hit an item box
        if (state['attachment'] != Attachment.BUBBLEGUM_SHIELD) and (len(items_to_collect) > 0):
            d = np.linalg.norm(from_polar_to_cartesian(state['items_position'][items_to_collect[0][0]]))
            if (state['items_type'][items_to_collect[0][0]] == ItemType.BONUS_BOX) and (speed > 0) and (((d - ExpertAgent.KART_LENGTH / 2.0 - ExpertAgent.DISTANCE_ASIDE_TO_AVOID_ITEM) / speed) < self.env.config.step_size * 2.5):
                action['fire'] = True
                return
        # Use bubblegum if the any kart behind is 'close' and straight behind
        straight_behind = False
        for kart_id, kart_pos in enumerate(state['karts_position']):
            if state['karts_rel_rank'][kart_id] >= 0:
                continue  # only consider karts behind
            behind_position = from_polar_to_cartesian(kart_pos)
            angle = get_xz_angle_between(np.array([0.0, 0.0, -1.0]), behind_position)
            if abs(angle) < 0.2:
                straight_behind = True
            if straight_behind and (np.linalg.norm(behind_position) < 8.0):
                action['fire'] = True
                return

    def __handle_cake(self, state: State, action: Action):
        """Handles cake powerup logic based on the current state.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
        """
        for kart_id, rel_rank in enumerate(state['karts_rel_rank']):
            if rel_rank <= 0:
                continue  # only consider karts ahead
            kart_ahead_id = kart_id
            # Do not fire if the target is too far
            kart_ahead_is_close = (state['karts_distance'][kart_ahead_id] < 20.0)
            straight_ahead = False
            rather_ahead = False  # used to not fire in big curves

            kart_ahead_has_gum = (state['karts_attachment'][kart_ahead_id] == Attachment.BUBBLEGUM_SHIELD)
            kart_ahead_position = from_polar_to_cartesian(state['karts_position'][kart_ahead_id])
            angle = get_xz_angle_between(np.array([0.0, 0.0, 1.0]), kart_ahead_position)
            if abs(angle) < 0.2:
                straight_ahead = True
            if abs(angle) < 0.5:
                rather_ahead = True
            # Do not fire if the kart is driving too slow
            speed = state['velocity'][2]
            kart_ahead_is_slow = (state['karts_velocity'][kart_ahead_id][2] < speed)
            # Establish if we should fire
            fire_ahead = 0.0
            if kart_ahead_is_close and rather_ahead:
                fire_ahead += 30.0 - state['karts_distance'][kart_ahead_id]
            else:
                fire_ahead -= 100.0
            # Don't fire at a kart that is slower than us. Reason is that
            # we can either save the cake for later since we will overtake
            # the kart anyway, or that this might force the kart ahead to
            # use its nitro/zipper (and then we will shoot since then the
            # kart is faster).
            if kart_ahead_is_slow:
                fire_ahead -= 50.0
            else:
                fire_ahead += 25.0
            # Try to take out a kart which has a swatter in priority
            kart_ahead_has_swatter = (state['karts_attachment'][kart_ahead_id] == Attachment.SWATTER)
            # If it is slower, the swatter is more dangerous
            if kart_ahead_has_swatter:
                if kart_ahead_is_slow:
                    fire_ahead += 75.0
                else:
                    fire_ahead += 15.0
            # Don't fire on a kart straight ahead with a bubblegum shield
            if kart_ahead_has_gum and straight_ahead:
                fire_ahead = -100.0
            # Finally decide to fire or not
            if fire_ahead > 0.0:
                action['fire'] = True
                return

    def __handle_bowling(self, state: State, action: Action):
        """Handles bowling powerup logic based on the current state.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
        """
        # Leave more time between bowling balls, since they are
        # slower, so it should take longer to hit something which
        # can result in changing our target.
        if state['time_since_last_fire'] < 2.0:
            return
        # Try to hit any kart ahead
        for kart_id, rel_rank in enumerate(state['karts_rel_rank']):
            if rel_rank <= 0:
                continue  # only consider karts ahead
            kart_ahead_id = kart_id
            # Consider angle towards karts
            straight_ahead = False
            # Get position of the kart ahead
            kart_ahead_position = from_polar_to_cartesian(state['karts_position'][kart_ahead_id])
            # Get angle towards that position
            angle = get_xz_angle_between(np.array([0.0, 0.0, 1.0]), kart_ahead_position)
            if abs(angle) < 0.2:
                straight_ahead = True
            # Don't fire on a kart straight ahead with a bubblegum shield
            kart_ahead_has_gum = (state['karts_attachment'][kart_ahead_id] == Attachment.BUBBLEGUM_SHIELD)
            if kart_ahead_has_gum and straight_ahead:
                return
            # Fire if the kart ahead is close enough
            dist_to_kart = np.linalg.norm(kart_ahead_position).item()
            fire = (straight_ahead and (dist_to_kart < 10.0))
            if fire:
                action['fire'] = True
                return

    def __handle_switch(self, state: State, action: Action, items_to_avoid: list[tuple[int, int]]):
        """Handles switch powerup logic based on the current state.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
            items_to_avoid (list[tuple[int, int]]): List to store item IDs to avoid and their associated path IDs.
        """
        # Get current speed
        speed = state['velocity'][2]
        # If there are items to avoid, and we are going towards them and close enough, use the switch
        if len(items_to_avoid) > 0:
            for (item_id, path_id) in items_to_avoid:
                item_type = state['items_type'][item_id]
                if (item_type == ItemType.BANANA) or (item_type == ItemType.BUBBLEGUM):
                    pos = from_polar_to_cartesian(state['items_position'][item_id])
                    angle = get_xz_angle_between(np.array([0.0, 0.0, 1.0]), pos)
                    # If we would reach the item soon (within 4 steps), use the switch
                    dist_to_item = np.linalg.norm(pos).item()
                    if (abs(angle) < 0.3) and (speed > 0) and ((dist_to_item - ExpertAgent.DISTANCE_ASIDE_TO_AVOID_ITEM) / speed < self.env.config.step_size * 4):
                        action['fire'] = True
                        return

    def __handle_swatter(self, state: State, action: Action, items_to_collect: list[tuple[int, int]]):
        """Handles swatter powerup logic based on the current state.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
            items_to_collect (list[tuple[int, int]]): List to store item IDs to collect and their associated path IDs.
        """
        # Get current attachment
        attachment = state['attachment']
        # Use swatter to remove bad attachments
        if (attachment == Attachment.PARACHUTE) or (attachment == Attachment.ANVIL):
            action['fire'] = True
            return
        # if it is a bomb, wait : we may pass it to another kart before the timer runs out
        # but if we may collect a bonus box soon, use it now
        speed = state['velocity'][2]
        has_bonus_box_ahead = False
        for (item_id, path_id) in items_to_collect:
            if state['items_type'][item_id] == ItemType.BONUS_BOX:
                pos = from_polar_to_cartesian(state['items_position'][item_id])
                angle = get_xz_angle_between(np.array([0.0, 0.0, 1.0]), pos)
                dist_to_item = np.linalg.norm(pos).item()
                if (abs(angle) < 0.3) and (speed > 0) and ((dist_to_item - ExpertAgent.DISTANCE_ASIDE_TO_AVOID_ITEM) / speed < self.env.config.step_size * 3):
                    has_bonus_box_ahead = True
                    break
        if has_bonus_box_ahead or ((attachment == Attachment.BOMB) and (state['attachment_time_left'] < 10.0)):
            action['fire'] = True
            return
        # Check if there are any karts within the swatter range
        any_kart_in_range = False
        for kart_pos in state['karts_position']:
            pos = from_polar_to_cartesian(kart_pos)
            dist_to_kart = np.linalg.norm(pos).item()
            if dist_to_kart < ExpertAgent.SWATTER_RADIUS:
                any_kart_in_range = True
                break
        if any_kart_in_range:
            action['fire'] = True
            return

    def __handle_collection_and_avoidance(self, state: State, target_point: np.ndarray, items_to_collect: list[tuple[int, int]], items_to_avoid: list[tuple[int, int]]):
        """Handles item collection and avoidance logic (avoid bad items and opponent karts) based on the current state.

        Args:
            state (State): The current state of the environment.
            target_point (np.ndarray): The target point to steer towards.
            items_to_collect (list[tuple[int, int]]): List to store item IDs to collect and their associated path IDs.
            items_to_avoid (list[tuple[int, int]]): List to store item IDs to avoid and their associated path IDs.
        """
        # Get current speed
        speed = state['velocity'][2]
        # 1) If there are items to avoid, and we are going towards them, avoid them
        if self.avoid_bad_items and (len(items_to_avoid) > 0) and (speed > 0):
            for (item_id, path_id) in items_to_avoid:
                item_to_avoid_position = from_polar_to_cartesian(state['items_position'][item_id])
                angle = get_xz_angle_between(np.array([0.0, 0.0, 1.0]), item_to_avoid_position)
                if abs(angle) < 0.3:
                    dist_to_item = np.linalg.norm(item_to_avoid_position).item()
                    nb_max_steps = 4
                    nb_steps_to_reach = ((dist_to_item - ExpertAgent.DISTANCE_ASIDE_TO_AVOID_ITEM - ExpertAgent.KART_LENGTH / 2.0) / speed) / self.env.config.step_size
                    if nb_steps_to_reach < nb_max_steps:
                        # Compute two possible avoidance target points aside the item (left and right)
                        avoid_target_point_1 = item_to_avoid_position - np.array([ExpertAgent.DISTANCE_ASIDE_TO_AVOID_ITEM + ExpertAgent.KART_WIDTH / 2.0 + 2.0, 0.0, 0.0])
                        avoid_target_point_2 = item_to_avoid_position + np.array([ExpertAgent.DISTANCE_ASIDE_TO_AVOID_ITEM + ExpertAgent.KART_WIDTH / 2.0 + 2.0, 0.0, 0.0])
                        # Choose the avoidance target point that is closer to the original target point
                        angle1 = get_xz_angle_between(np.array([0.0, 0.0, 1.0]), avoid_target_point_1)
                        angle2 = get_xz_angle_between(np.array([0.0, 0.0, 1.0]), avoid_target_point_2)
                        if abs(angle1) < abs(angle2):
                            return avoid_target_point_1
                        else:
                            return avoid_target_point_2
        # 2) If there are karts that may collide with us, avoid them
        # But if our kart has a bomb attached to it, we won't try to avoid other karts
        # as we may want to hit them on purpose to pass the bomb to them
        if self.avoid_karts and (state['attachment'] != Attachment.BOMB):
            for i, kart_pos in enumerate(state['karts_position']):
                if state['karts_rel_rank'][i] < 0:
                    continue  # Only consider karts ahead
                kart_speed = state['karts_velocity'][i][2]
                relative_speed = speed - kart_speed
                if relative_speed <= 0:
                    continue  # Only consider karts that are slower than us
                kart_position = from_polar_to_cartesian(kart_pos)
                angle = get_xz_angle_between(np.array([0.0, 0.0, 1.0]), kart_position)
                if abs(angle) < 0.3:
                    dist_to_kart = np.linalg.norm(kart_position).item()
                    nb_max_steps = 4
                    nb_steps_to_reach = ((dist_to_kart - ExpertAgent.KART_LENGTH) / relative_speed) / self.env.config.step_size
                    if nb_steps_to_reach < nb_max_steps:
                        # Compute two possible avoidance target points aside the kart (left and right)
                        avoid_target_point_1 = kart_position - np.array([ExpertAgent.KART_WIDTH + 2.0, 0.0, 0.0])
                        avoid_target_point_2 = kart_position + np.array([ExpertAgent.KART_WIDTH + 2.0, 0.0, 0.0])
                        # Choose the avoidance target point that is closer to the original target point
                        angle1 = get_xz_angle_between(np.array([0.0, 0.0, 1.0]), avoid_target_point_1)
                        angle2 = get_xz_angle_between(np.array([0.0, 0.0, 1.0]), avoid_target_point_2)
                        if abs(angle1) < abs(angle2):
                            return avoid_target_point_1
                        else:
                            return avoid_target_point_2
        # 3) If there's nothing to avoid, try to collect bonuses if close enough
        # TODO: this logic is still experimental and must be improved
        if self.collect_items and (len(items_to_collect) > 0) and (speed > 0):
            for (item_id, path_id) in items_to_collect:
                item_to_collect_position = from_polar_to_cartesian(state['items_position'][item_id])
                if abs(item_to_collect_position[0]) < 2.0:
                    return target_point  # already aligned, no need to adjust
                else:
                    dist_to_item = np.linalg.norm(item_to_collect_position).item()
                    nb_max_steps = 4
                    nb_steps_to_reach = ((dist_to_item - ExpertAgent.DISTANCE_ASIDE_TO_AVOID_ITEM - ExpertAgent.KART_LENGTH / 2.0) / speed) / self.env.config.step_size
                    if nb_steps_to_reach < nb_max_steps:
                        return item_to_collect_position + np.array([2.0, 0.0, 0.0]) * np.sign(item_to_collect_position[0])
        # Return the new target point
        return target_point

    def __steer_to_point(self, target_point: np.ndarray):
        """Calculates the steer angle needed to steer towards a target point.

        Args:
            target_point (np.ndarray): The target point to steer towards.

        Returns:
            float: The steer angle in the range [-pi, pi].
        """
        # The center of the rotation can be computed by the fact that the distance
        # to the kart and to the point to aim at must be the same:
        # r**2 = (x - r)**2 + z**2
        # where r is the radius (= position on the X axis), and x, z are the
        # local coordinates of the point to aim at. Solving for r
        # results in r = (x**2 + y**2) / (2 * x)

        # Catch edge case where target_point[0] is zero to avoid division by zero
        if target_point[0] == 0:
            return 0.0
        # Otherwise
        radius = (target_point[0] * target_point[0] + target_point[2] * target_point[2]) / (2.0 * target_point[0])
        # Compute steer angle
        steer_angle = np.atan(ExpertAgent.WHEEL_BASE / radius).item()
        # Return the steer angle
        return steer_angle

    def __set_steering(self, state: State, action: Action, steer_angle: float, track_direction: DirectionType,
                       last_path_id: Optional[int], curve_center: Optional[np.ndarray], curve_radius: Optional[float]):
        """Sets the steering action based on the steer angle.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
            steer_angle (float): The steer angle in radians.
            track_direction (DirectionType): The direction type of the track.
            last_path_id (Optional[int]): The index of the last path segment considered.
            curve_center (Optional[np.ndarray]): The center of the turn (as a 3D point).
            curve_radius (Optional[float]): The radius of the turn.
        """
        # Calculate the steer fraction
        steer_fraction = steer_angle / state['max_steer_angle']
        # If drifting is enabled, determine if we can drift
        if self.enable_drift and (not state['stuck']):
            can_drift = self.__can_drift(state, steer_fraction, track_direction, last_path_id, curve_center, curve_radius)
            action['drift'] = can_drift
            # If we are drifting, adjust the steering accordingly
            if can_drift and state['skid_direction'] != 0:
                steer_fraction = self.__get_steering_when_drifting(state, steer_fraction)
        # Set the steer action
        action['steer'] = np.clip(steer_fraction, -1.0, 1.0).item()

    def __determine_track_direction(self, state: State, target_position: np.ndarray):
        """Determines the direction of the track based on the upcoming path segments.

        Args:
            state (State): The current state of the environment.
            target_position (np.ndarray): The target position to aim for.

        Returns:
            DirectionType: The direction type of the track.
        """
        # Get the maximum number of path segments to consider
        max_nb_paths = len(state['paths_start'])
        assert max_nb_paths >= 4, "There should be at least 4 path segments to determine the track direction"
        # Get the center point of the upcoming path segment
        next_path_center = (from_polar_to_cartesian(state['paths_start'][1]) + from_polar_to_cartesian(state['paths_end'][1])) / 2.0
        # Get velocity vector
        velocity = from_polar_to_cartesian(state['velocity'])
        # Compute the angle between the velocity and the next path center
        velocity_to_next_angle = get_xz_angle_between(velocity, next_path_center)
        # If the angle is quite big, we consider the direction to be undefined to avoid
        # taking wrong decisions until we get back on track
        if abs(velocity_to_next_angle) > 0.3 * np.pi:
            return DirectionType.DIR_UNDEFINED, None
        # Determine if the track is going left, right or straight
        # Get the first angle (between direction to next path and its center direction to the path next to it)
        current_path_center = next_path_center
        next_path_center = (from_polar_to_cartesian(state['paths_start'][2]) + from_polar_to_cartesian(state['paths_end'][2])) / 2.0
        v1 = target_position
        v2 = next_path_center - current_path_center
        angle1 = get_xz_angle_between(v1, v2)
        # Get second angle (between path center direction and the next path center direction)
        current_path_center = next_path_center
        next_path_center = (from_polar_to_cartesian(state['paths_start'][3]) + from_polar_to_cartesian(state['paths_end'][3])) / 2.0
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
            next_path_center = (from_polar_to_cartesian(state['paths_start'][i + 1]) + from_polar_to_cartesian(state['paths_end'][i + 1])) / 2.0
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

    def __determine_turn_radius(self, state: State, last_path_id: Optional[int]):
        """Determine the center and radius of the turn based on the last path segment.

        Args:
            state (State): The current state of the environment.
            last_path_id (Optional[int]): The index of the last path segment considered.

        Returns:
            tuple[Optional[np.ndarray], Optional[float]]: The center of the turn (as a 3D point) and the radius of the turn.
        """
        # If last_path_id is None, we cannot determine the turn radius
        if last_path_id is None:
            return None, None
        # Get the center point of the last path segment
        last_path_center = (from_polar_to_cartesian(state['paths_start'][last_path_id]) + from_polar_to_cartesian(state['paths_end'][last_path_id])) / 2.0
        # Compute the mid point and orthogonal vector
        mid = 0.5 * last_path_center
        direction = last_path_center
        orthogonal = np.array([direction[2], 0.0, -direction[0]])
        q1 = mid + orthogonal
        # Compute line equation between mid and q1
        line1_a = (q1[2] - mid[2]) / (q1[0] - mid[0]) if (q1[0] - mid[0]) != 0 else None
        line1_b = mid[2] - line1_a * mid[0] if line1_a is not None else None
        # Compute line intersection with x axis
        if (line1_a is not None) and (line1_a != 0):
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

    def __can_drift(self, state: State, steer_fraction: float, track_direction: DirectionType,
                    last_path_id: Optional[int], curve_center: Optional[np.ndarray], curve_radius: Optional[float]):
        """Determines if the kart can perform a drift based on its current speed and track conditions.

        Args:
            state (State): The current state of the environment.
            steer_fraction (float): The fraction of the maximum steering angle being used.
            track_direction (DirectionType): The direction type of the track.
            last_path_id (Optional[int]): The index of the last path segment considered.
            curve_center (Optional[np.ndarray]): The center of the turn (as a 3D point).
            curve_radius (Optional[float]): The radius of the turn.

        Returns:
            bool: True if the kart can drift, False otherwise.
        """
        # 1) Track changes
        # If the kart is off-road, we cannot drift
        if not state['is_on_road']:
            return False
        # If the track is straight or undefined, we cannot drift
        if (track_direction == DirectionType.DIR_STRAIGHT) or (track_direction == DirectionType.DIR_UNDEFINED):
            return False
        # If the kart is already skidding, and the track changed direction, we cannot drift
        if ((state['skid_direction'] == 1) and (track_direction == DirectionType.DIR_LEFT)) or ((state['skid_direction'] == -1) and (track_direction == DirectionType.DIR_RIGHT)):
            return False
        # 2) Speed condition
        # Get current speed
        speed = state['velocity'][2]
        # Only try to drift if we are above the minimum speed
        if speed < ExpertAgent.SKID_MIN_SPEED:
            return False
        # 3) Conditions to continue drifting
        # Estimate how long it takes to finish the curve
        assert (last_path_id is not None) and (curve_center is not None) and (curve_radius is not None), "last_path_id, curve_center and curve_radius should not be None here"
        last_path_center = (from_polar_to_cartesian(state['paths_start'][last_path_id]) + from_polar_to_cartesian(state['paths_end'][last_path_id])) / 2.0
        diff_kart = -curve_center
        diff_last = last_path_center - curve_center
        angle = get_xz_angle_between(diff_kart, diff_last)
        length = curve_radius * abs(angle)
        duration = length / speed
        # If we are already drifting, continue to drift until we reached a little bit before the end of the curve
        already_drifting = state['prev_actions'][-1]['drift'] and (state['skid_direction'] != 0)
        if already_drifting and (duration > 1.0):
            # And check if the steer fraction is whithin the allowed range
            if state['skid_direction'] == 1:  # drifting right
                if steer_fraction < ExpertAgent.SKID_REDUCE_TURN_MIN - 0.15:
                    return False
            else:  # drifting left
                if steer_fraction > -ExpertAgent.SKID_REDUCE_TURN_MIN + 0.15:
                    return False
            # else we can continue drifting
            return True
        # 4) Conditions to start drifting
        # We want to start drifting so we check if we are steering left in a right curve or steering right in a left curve, if it's the case we cannot drift
        if ((steer_fraction > 0) and (track_direction == DirectionType.DIR_LEFT)) or ((steer_fraction < 0) and (track_direction == DirectionType.DIR_RIGHT)):
            return False
        # If the time to finish the curve is higher than the minimum skid time till bonus plus 1 second to recover from the drift, we should drift
        should_drift = duration > (ExpertAgent.SKID_TIME_TILL_BONUS[0] + 1.0)
        if should_drift:
            # Finally check if the steer fraction is whithin the allowed range
            if track_direction == DirectionType.DIR_RIGHT:
                if steer_fraction < ExpertAgent.SKID_REDUCE_TURN_MIN:
                    return False
            else:  # track_direction == DirectionType.DIR_LEFT
                if steer_fraction > -ExpertAgent.SKID_REDUCE_TURN_MIN:
                    return False
            # else we can drift
            return True
        else:
            # If we shouldn't drift, we cannot drift
            return False

    def __get_steering_when_drifting(self, state: State, steering: float):
        """Calculates the adjusted steering value when the kart is drifting.

        Args:
            state (State): The current state of the environment.
            steering (float): The original steering value.

        Returns:
            float: The adjusted steering value for drifting.
        """
        if state['time_since_skid_start'] < 3 * self.env.config.step_size:
            return steering  # no adjustment on the first frames of drifting
        delta = ExpertAgent.SKID_REDUCE_TURN_MAX - ExpertAgent.SKID_REDUCE_TURN_MIN
        if state['skid_direction'] == 0:  # not skidding
            return steering
        elif state['skid_direction'] == 1:  # skidding right
            f = (steering - ExpertAgent.SKID_REDUCE_TURN_MIN) / delta
            return 2.0 * f - 1.0
        else:  # skidding left
            f = (steering + ExpertAgent.SKID_REDUCE_TURN_MIN) / delta
            return 2.0 * f + 1.0

    def __estimate_finish_time(self, state: State):
        """Estimates the time remaining to finish the race based on the current state."""
        # Get full distance of the race
        track_length = self.env.track.length
        full_distance = self.env.config.laps * track_length
        # Get full distance covered by the kart
        kart_overall_distance = state['overall_distance_ratio'] * full_distance
        # Get average speed of the kart over all the race so far
        average_speed = 1.0 if state['game_time'] == 0 else (kart_overall_distance / state['game_time'])
        # Calculate estimated finish time
        est_time = state['game_time'] + (full_distance - kart_overall_distance) / average_speed
        # Return the estimated finish time
        return est_time

    def __evaluate_items(self, state: State, item_id: int, current_path_id: int, items_to_collect: list[tuple[int, int]], items_to_avoid: list[tuple[int, int]]):
        """Evaluates items to collect and avoid based on the kart's aim direction.

        Args:
            state (State): The current state of the environment.
            item_id (int): The ID of the item to evaluate.
            current_path_id (int): The ID of the current path segment.
            items_to_collect (list[tuple[int, int]]): List to store item IDs to collect and their associated path IDs.
            items_to_avoid (list[tuple[int, int]]): List to store item IDs to avoid and their associated path IDs.
        """
        item_position = from_polar_to_cartesian(state['items_position'][item_id])
        item_type = state['items_type'][item_id]

        avoid = False
        # Negative items to avoid
        if item_type == ItemType.BUBBLEGUM:
            avoid = True
        elif item_type == ItemType.BANANA:
            avoid = True
        # Positive items to collect
        elif item_type == ItemType.NITRO_BIG:
            # Only collect nitro, if it can actually be stored.
            if state['energy'] == ExpertAgent.NITRO_MAX:
                return
        elif item_type == ItemType.NITRO_SMALL:
            # Only collect nitro, if it can actually be stored.
            if state['energy'] == ExpertAgent.NITRO_MAX:
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
            other_item_position = from_polar_to_cartesian(state['items_position'][idx])
            other_item_distance = np.linalg.norm(other_item_position).item()
            if item_distance < other_item_distance:
                break
            insert_index += 1

        target_list.insert(insert_index, (item_id, current_path_id))

    def __check_if_pos_in_path(self, state: State, position: np.ndarray, path_id: int):
        """Checks if a given position is within the path segment.

        Args:
            state (State): The current state of the environment.
            position (np.ndarray): The position to check.
            path_id (int): The ID of the path segment.

        Returns:
            bool: True if the position is within the path segment, False otherwise.
        """
        path_start = from_polar_to_cartesian(state['paths_start'][path_id])
        path_end = from_polar_to_cartesian(state['paths_end'][path_id])
        path_dir = path_end - path_start
        path_length = np.linalg.norm(path_dir)
        if path_length == 0:
            return False
        path_dir /= path_length
        rel_pos = position - path_start
        proj_length = np.dot(rel_pos, path_dir)
        proj_width = np.linalg.norm(rel_pos - proj_length * path_dir)
        half_path_width = 0.5 * state['paths_width'][path_id]
        return (0 <= proj_length) and (proj_length <= path_length) and (proj_width <= half_path_width - 0.5 * ExpertAgent.KART_WIDTH)
