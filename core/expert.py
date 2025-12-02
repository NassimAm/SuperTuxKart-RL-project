from core.state import State, Phase, Attachment, Powerup, ItemType
from core.action import Action
from core.utils import from_polar_to_cartesian, get_xz_angle_between
from typing import Optional
import numpy as np
from enum import Enum


class DirectionType(Enum):
    """To indicate in which direction the track is going: straight, left, right. The undefined direction is used by the AI only."""
    DIR_STRAIGHT = 0
    DIR_LEFT = 1
    DIR_RIGHT = 2
    DIR_UNDEFINED = 3


class ExpertAgent:
    """An expert agent for SuperTuxKart that uses predefined strategies to navigate the track. This implementation is an
    attempt to replicate the same logic as the original C++ expert agent from SuperTuxKart. The original code that we replicate from
    is found here https://github.com/bpiwowar/pystk2/blob/master/src/karts/controller/skidding_ai.cpp"""

    enable_drift: bool
    """Whether to enable drifting behavior."""

    # ============================== Path characteristics ==============================
    MAX_STRAIGHT_ANGLE = 0.1  # radians
    # ============================== Kart characteristics ==============================
    # NOTE: These values assume that we always have the same type of kart (the default penguin kart)
    # see https://github.com/bpiwowar/pystk2/blob/f86f21efbf137c95c26f85d113dd1525a8fc4ea6/data/kart_characteristics.xml for more details
    TURN_RADIUS = {0: 2.3, 10: 8.625, 25: 17.25, 45: 34.5}
    # nitro
    NITRO_DURATION = 1.0  # seconds
    NITRO_FADE_OUT_TIME = 2.4  # seconds
    NITRO_MAX_SPEED_INCREASE = 5.0  # m/s
    NITRO_CONSUMPTION = 1.25
    NITRO_SMALL_CONTAINER = 1.0
    NITRO_BIG_CONTAINER = 3.0
    NITRO_MAX = 20.0
    # zipper
    ZIPPER_DURATION = 3.5  # seconds
    ZIPPER_FADE_OUT_TIME = 1.0  # seconds
    ZIPPER_MAX_SPEED_INCREASE = 15.0  # m/s
    # skidding
    SKID_BONUS_SPEED = [4.5, 6.5]
    SKID_TIME_TILL_BONUS = [1.0, 3.0]
    SKID_REDUCE_TURN_MIN = 0.2
    SKID_REDUCE_TURN_MAX = 0.8
    SKID_MIN_SPEED = 10  # m/s
    # parachute
    PARACHUTE_DURATION_OTHER = 2.5  # seconds
    # swatter
    SWATTER_DISTANCE = 9.0  # meters ** 2
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
    SHIELD_RESTRICT_WEAPONS = False
    # ============================== Personal Assumptions ==============================
    # NOTE: These values are personal assumptions made to simplify the implementation
    MIN_DISTANCE_TO_AVOID_ITEM = 1.0  # meters
    MAX_TIME_SINCE_LAST_FIRE = 10.0  # seconds

    def __init__(self, enable_drift: bool = True):
        """Initializes the ExpertAgent.

        Args:
            enable_drift (bool): Whether to enable drifting behavior. Default is True.
        """
        self.enable_drift = enable_drift

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
            action['rescue'] = True
            # TODO: add more expert logic here
            return Action.to_env(action)
        # If the race didn't start yet, handle that special case
        if state['phase'].value < Phase.RACE_PHASE.value:
            self.__handle_race_start(state, action)
            return Action.to_env(action)
        # Determine the track direction
        track_direction, last_path_id = self.__determine_track_direction(state)
        print("Track direction:", track_direction, "last path id:", last_path_id)
        # Get track curve
        curve_center, curve_radius = self.__determine_turn_radius(state, last_path_id)
        print("Curve center:", curve_center, "Curve radius:", curve_radius)
        # Handle acceleration and braking
        self.__handle_acceleration_and_braking(state, action)
        # Handle steering
        self.__handle_steering(state, action, track_direction, last_path_id, curve_center, curve_radius)
        # Return the action in the expected environment format
        print("Action:", action)
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

    def __handle_acceleration_and_braking(self, state: State, action: Action):
        """Handles acceleration and braking logic based on the current state.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
        """
        action['acceleration'] = 1.0

    def __handle_steering(self, state: State, action: Action, track_direction: DirectionType, last_path_id: Optional[int],
                          curve_center: Optional[np.ndarray], curve_radius: Optional[float]):
        """Handles steering logic based on the current state.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
            track_direction (DirectionType): The direction type of the track.
            last_path_id (Optional[int]): The index of the last path segment considered.
            curve_center (Optional[np.ndarray]): The center of the turn (as a 3D point).
            curve_radius (Optional[float]): The radius of the turn.
        """
        # Set the center of the next path segment as a default target point
        target_position = (from_polar_to_cartesian(state['paths_start'][2]) + from_polar_to_cartesian(state['paths_end'][2])) / 2.0
        # Compute the required steer to reach the target point
        steer_angle = self.__steer_to_point(state, target_position)
        # Calculate the steer fraction
        steer_fraction = steer_angle / state['max_steer_angle']
        # If drifting is enabled, determine if we can drift
        if self.enable_drift:
            can_drift = self.__can_drift(state, steer_fraction, track_direction, last_path_id, curve_center, curve_radius)
            action['drift'] = can_drift
            # If we are drifting, adjust the steering accordingly
            if can_drift and state['skid_direction'] != 0:
                steer_fraction = self.__get_steering_when_drifting(state, steer_fraction)
        # Set the steer action
        action['steer'] = np.clip(steer_fraction, -1.0, 1.0).item()

    def __steer_to_point(self, state: State, target_point: np.ndarray):
        """Calculates the steer angle needed to steer towards a target point.

        Args:
            state (State): The current state of the environment.
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
        radius = (target_point[0] * target_point[0] + target_point[2] * target_point[2]) / (2.0 * target_point[0])
        # Compute steer angle
        steer_angle = np.atan(ExpertAgent.WHEEL_BASE / radius).item()
        # Return the steer angle
        return steer_angle

    def __determine_track_direction(self, state: State):
        """Determines the direction of the track based on the upcoming path segments.

        Args:
            state (State): The current state of the environment.

        Returns:
            DirectionType: The direction type of the track.
        """
        # Get the maximum number of path segments to consider
        max_nb_paths = len(state['paths_start'])
        assert max_nb_paths >= 4, "There should be at least 4 path segments to determine the track direction"
        # Get the center point of the upcoming path segment
        next_path_center = (from_polar_to_cartesian(state['paths_start'][1]) + from_polar_to_cartesian(state['paths_end'][1])) / 2.0
        # Get velocity vector
        velocity = state['velocity']
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
        next_path_center = (from_polar_to_cartesian(state['paths_start'][2]) + from_polar_to_cartesian(state['paths_end'][2])) / 2.0
        v1 = current_path_center
        v2 = next_path_center - current_path_center
        angle1 = get_xz_angle_between(v1, v2)
        # Get second angle (between path center direction and the next path center direction)
        current_path_center = next_path_center
        next_path_center = (from_polar_to_cartesian(state['paths_start'][3]) + from_polar_to_cartesian(state['paths_end'][3])) / 2.0
        v1 = v2
        v2 = next_path_center - current_path_center
        angle2 = get_xz_angle_between(v1, v2)
        rel_angle = angle1 + angle2
        print("Initial relative angle:", rel_angle)
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
            print("New relative angle:", new_rel_angle)
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
        # If the track is straight or undefined, we cannot drift
        if (track_direction == DirectionType.DIR_STRAIGHT) or (track_direction == DirectionType.DIR_UNDEFINED):
            return False
        # Get current speed
        speed = np.linalg.norm(state['velocity']).item()
        # Only try to drift if we are above the minimum speed
        if speed < ExpertAgent.SKID_MIN_SPEED:
            return False
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
        if already_drifting and (duration < 0.5):
            return True
        # We want to start drifting so we check if we are steering left in a right curve or steering right in a left curve, if it's the case we cannot drift
        if ((steer_fraction > 0) and (track_direction == DirectionType.DIR_LEFT)) or ((steer_fraction < 0) and (track_direction == DirectionType.DIR_RIGHT)):
            return False
        # If the time to finish the curve is higher than the minimum skid time till bonus, we should drift
        return duration >= ExpertAgent.SKID_TIME_TILL_BONUS[0]

    def __get_steering_when_drifting(self, state: State, steering: float):
        """Calculates the adjusted steering value when the kart is drifting.

        Args:
            state (State): The current state of the environment.
            steering (float): The original steering value.

        Returns:
            float: The adjusted steering value for drifting.
        """
        delta = ExpertAgent.SKID_REDUCE_TURN_MAX - ExpertAgent.SKID_REDUCE_TURN_MIN
        if state['skid_direction'] == 0:  # not skidding
            return steering
        elif state['skid_direction'] == 1:  # skidding right
            f = (steering - ExpertAgent.SKID_REDUCE_TURN_MIN) / delta
            return 2.0 * f - 1.0
        else:  # skidding left
            f = (steering + ExpertAgent.SKID_REDUCE_TURN_MIN) / delta
            return 2.0 * f + 1.0
