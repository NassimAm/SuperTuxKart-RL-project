import numpy as np
from enum import Enum
from typing import TypedDict, Optional
from core.state import State, Phase, Attachment, Powerup, ItemType
from core.action import Action
import json
from gymnasium import Env
from pystk2_gymnasium.envs import STKRaceEnv
import random
from core.utils import from_polar_to_cartesian, get_xz_angle_between


class CrashType:
    """Class to discribe the type of crash that happened to the agent"""

    m_road: bool
    """true if we are going to 'crash' with the bounds of the road"""
    m_kart: int
    """-1 if no crash, pos numbers are the kart it crashes with"""

    def __init__(self, m_road: bool = False, m_kart: int = -1):
        self.road = m_road
        self.kart = m_kart

    def reset(self):
        self.road = False
        self.kart = -1


class DirectionType(Enum):
    """To indicate in which direction the track is going: straight, left, right. The undefined direction is used by the AI only."""
    DIR_STRAIGHT = 0
    DIR_LEFT = 1
    DIR_RIGHT = 2
    DIR_UNDEFINED = 3


class SkillType(Enum):
    """This is used by computeSkill to know what skill is used"""
    ITEM_SKILL = 0
    NITRO_SKILL = 1


class ExpertAgentConfig(TypedDict):
    time_full_steer: float
    straight_length_for_zipper: int
    use_slipstream: bool
    disable_slipstream_usage: bool
    shield_incoming_radius: int
    false_start_probability: float
    min_start_delay: float
    max_start_delay: float
    nitro_usage: int
    item_skill: int
    collect_avoid_items: bool
    handle_bomb: bool
    first_speed_cap: str
    last_speed_cap: str
    max_item_angle: float
    max_item_angle_high_speed: float
    bad_item_closeness: int
    collect_item_probability: str
    rb_skid_probability: str
    skidding_threshold: float

    def load_from_json(path: str):
        with open(path, 'r') as file:
            data = json.load(file)
        return ExpertAgentConfig(
            time_full_steer=data["time-full-steer"],
            straight_length_for_zipper=data["straight-length-for-zipper"],
            use_slipstream=data["use-slipstream"],
            disable_slipstream_usage=data["disable-slipstream-usage"],
            shield_incoming_radius=data["shield-incoming-radius"],
            false_start_probability=data["false-start-probability"],
            min_start_delay=data["min-start-delay"],
            max_start_delay=data["max-start-delay"],
            nitro_usage=data["nitro-usage"],
            item_skill=data["item-skill"],
            collect_avoid_items=data["collect-avoid-items"],
            handle_bomb=data["handle-bomb"],
            first_speed_cap=data["first-speed-cap"],
            last_speed_cap=data["last-speed-cap"],
            max_item_angle=data["max-item-angle"],
            max_item_angle_high_speed=data["max-item-angle-high-speed"],
            bad_item_closeness=data["bad-item-closeness"],
            collect_item_probability=data["collect-item-probability"],
            rb_skid_probability=data["rb-skid-probability"],
            skidding_threshold=data["skidding-threshold"]
        )


class ExpertAgent:
    """An expert agent for SuperTuxKart that uses predefined strategies to navigate the track. This implementation is an
    attempt to replicate the same logic as the original C++ expert agent from SuperTuxKart. The original code that we replicate from
    is found here https://github.com/bpiwowar/pystk2/blob/master/src/karts/controller/skidding_ai.cpp"""
    env: STKRaceEnv
    """The SuperTuxKart environment"""
    config: ExpertAgentConfig
    """Configuration options for the expert agent (these options control the difficulty of the agent)"""
    max_speed: float
    """The maximum speed of the kart based on the difficulty level"""

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

    def __init__(self, env: Env, config: ExpertAgentConfig, enable_drift: bool = True):
        """Initializes the ExpertAgent with default settings."""
        self.env = env.unwrapped
        self.config = config
        self.max_speed = ExpertAgent.MAX_SPEED * ExpertAgent.MAX_SPEED_FACTORS[env.unwrapped.difficulty]
        self.enable_drift = enable_drift

    def get_action(self, input_state: dict) -> np.ndarray:
        """Given the current state, returns the action to take as an expert agent.

        Args:
            state (np.ndarray): The current state of the environment.

        Returns:
            np.ndarray: The action to take.
        """
        # Convert flattened state to State object
        state: State = State.from_env(input_state)
        # Initialize action components
        action = Action(
            acceleration=0.0,
            brake=False,
            drift=False,
            fire=False,
            nitro=False,
            rescue=False,
            steer=0.0,
            look_back=False
        )
        # If the kart needs to be rescued, do it now (and nothing else)
        if state['stuck']:
            action['rescue'] = True
            return Action.to_env(action)
        # If the race didn't start yet, handle that special case
        if state['phase'].value < Phase.RACE_PHASE.value:
            self.__handle_race_start(state, action)
            return Action.to_env(action)
        # Compute nearest karts
        kart_ahead_id, kart_behind_id = self.__compute_nearest_karts(state)
        # Check for potential crashes
        crash_kart_id = self.__check_crashes(state)
        # Determine track direction
        track_direction, curve_center, curve_radius, last_path_id = self.__determine_track_direction(state)
        # Handle acceleration and braking
        self.__handle_acceleration_and_braking(state, action, curve_radius, track_direction,
                                               kart_ahead_id)
        # Handle steering
        self.__handle_steering(state, action, kart_ahead_id,
                               kart_behind_id, crash_kart_id,
                               dt=self.env.config.step_size,
                               current_track_direction=track_direction,
                               last_path_id=last_path_id,
                               curve_center=curve_center,
                               curve_radius=curve_radius)
        # Return the action in the environment format
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

    def __compute_nearest_karts(self, state: State):
        """Computes the nearest karts ahead and behind the agent's kart.

        Args:
            state (State): The current state of the environment.

        Returns:
            tuple[np.ndarray, np.ndarray]: The positions of the nearest kart ahead and behind the agent's kart.
        """
        karts_rel_rank = state['karts_rel_rank']
        kart_ahead_id = next((i for i in range(len(karts_rel_rank)) if karts_rel_rank[i] > 0), None)
        kart_behind_id = next((i for i in range(len(karts_rel_rank)) if karts_rel_rank[i] < 0), None)
        return kart_ahead_id, kart_behind_id

    def __get_angle_to_next_path(self, state: State, current_id: int):
        """Get the angle between the kart's velocity and the direction of the next path segment.

        Args:
            state (State): The current state of the environment.
            current_id (int): The index of the current path segment.
        """
        current_path_center = (from_polar_to_cartesian(state['paths_start'][current_id]) + from_polar_to_cartesian(state['paths_end'][current_id])) / 2.0
        next_path_center = (from_polar_to_cartesian(state['paths_start'][current_id + 1]) + from_polar_to_cartesian(state['paths_end'][current_id + 1])) / 2.0
        current_path_direction = from_polar_to_cartesian(state['paths_end'][current_id]) - from_polar_to_cartesian(state['paths_start'][current_id])
        center_direction = next_path_center - current_path_center
        angle_to_next = get_xz_angle_between(current_path_direction, center_direction)
        return angle_to_next

    def __determine_direction(self, state: State, current_path_id: int = 0):
        """Determine the direction of the track (straight, left, right) based on upcoming paths.

        Args:
            state (State): The current state of the environment.
            current_path_id (int): The index of the current path segment.

        Returns:
            tuple[DirectionType, int]: The determined direction type and the last path index considered before the direction changed or we hit the state path number limit.
        """
        state_paths = len(state['paths_start'])
        assert current_path_id + 2 < state_paths, "Not enough paths to determine direction change."
        # Define a threshold angle to consider "straight"
        max_straight_angle = 0.1
        angle_to_next = self.__get_angle_to_next_path(state, current_id=current_path_id)
        angle_to_next_next = self.__get_angle_to_next_path(state, current_id=current_path_id + 1)
        rel_angle = angle_to_next + angle_to_next_next
        last_path_id = current_path_id + 2
        if abs(rel_angle) < max_straight_angle:
            rel_angle = 0.0

        for i in range(2, state_paths - 1 - current_path_id):
            angle_to_next = angle_to_next_next
            angle_to_next_next = self.__get_angle_to_next_path(state, current_id=last_path_id)
            new_rel_angle = angle_to_next + angle_to_next_next
            if abs(new_rel_angle) < max_straight_angle:
                new_rel_angle = 0.0
            if (rel_angle != 0 and new_rel_angle == 0) or (rel_angle == 0 and new_rel_angle != 0) or (rel_angle * new_rel_angle < 0):
                break
            rel_angle = new_rel_angle
            last_path_id = current_path_id + i + 1

        return (DirectionType.DIR_STRAIGHT if rel_angle == 0.0 else (DirectionType.DIR_RIGHT if rel_angle > 0 else DirectionType.DIR_LEFT),
                last_path_id)

    def __determine_turn_radius(self, state: State, last_path_id: int):
        """Determine the center and radius of the turn based on the last path segment.

        Args:
            state (State): The current state of the environment.
            last_path_id (int): The index of the last path segment considered.

        Returns:
            tuple[np.ndarray, float]: The center of the turn (as a 3D point) and the radius of the turn.
        """
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
            curve_center = np.array([intersect_x, 0.0, intersect_z])
            curve_radius = np.linalg.norm(curve_center)
        else:
            # If there is no intersection, assume that the curve is a semi circle around the mid point
            curve_center = mid
            curve_radius = 0.5 * np.linalg.norm(last_path_center)
        # Return the curve center and radius
        return curve_center, curve_radius

    def __determine_track_direction(self, state: State):
        """Determine the direction of the track (straight, left, right) based on upcoming paths.

        Args:
            state (State): The current state of the environment.

        Returns:
            tuple[DirectionType, np.ndarray, float]: The determined direction type, the center of the turn (if any), and the radius of the turn (if any).
        """
        current_path_center = (from_polar_to_cartesian(state['paths_start'][0]) + from_polar_to_cartesian(state['paths_end'][0])) / 2.0
        next_path_center = (from_polar_to_cartesian(state['paths_start'][1]) + from_polar_to_cartesian(state['paths_end'][1])) / 2.0
        path_direction = next_path_center - current_path_center
        angle_to_track = 0.0
        velocity_norm = np.linalg.norm(state['velocity'])
        if velocity_norm > 1e-1:
            angle_to_track = get_xz_angle_between(state['velocity'], path_direction)
        # In certain circumstances (esp. S curves) it is possible that the
        # kart is not facing in the direction of the track. In this case
        # determining the curve radius based on the direction the kart is
        # facing results in very incorrect results (example: if the kart is
        # in a tight curve, but already facing towards the last point of the
        # curve - in this case a huge curve radius will be computes (since
        # the kart is nearly going straight), while in fact the kart would
        # go across the circle and not along, bumping into the track).
        # To avoid this we set the direction to undefined in this case,
        # which causes the kart to brake (which will allow the kart to
        # quicker be aligned with the track again).
        curve_center = None  # TODO: maybe set it to better value to avoid issues
        curve_radius = np.inf
        if abs(angle_to_track) > 0.22222 * np.pi:
            return DirectionType.DIR_UNDEFINED, curve_center, curve_radius, None

        path_direction_type, last_path_id = self.__determine_direction(state, current_path_id=1)

        if path_direction_type == DirectionType.DIR_RIGHT or path_direction_type == DirectionType.DIR_LEFT:
            curve_center, curve_radius = self.__determine_turn_radius(state, last_path_id)

        return path_direction_type, curve_center, curve_radius, last_path_id

    def __check_crashes(self, state: State):
        """Check for potential crashes with other karts.

        Args:
            state (State): The current state of the environment.

        Returns:
            Optional[int]: The index of the kart that will be crashed into, or None if no crash is detected.
        """
        # NOTE: The crashing logic was simplified compared to the C++ version (not considering road crashes)
        speed = np.linalg.norm(state['velocity'])
        steps = int(speed / ExpertAgent.KART_LENGTH)
        steps = min(max(steps, 2), 1000)
        # The AI drives significantly better with more steps, so for now
        # add 5 additional steps.
        steps += 5
        if speed == 0:
            return None

        velocity_norm = state['velocity'] / speed
        # Time it takes to drive for m_kart_length units.
        dt = ExpertAgent.KART_LENGTH / speed
        for i in range(1, steps):
            step_coord = velocity_norm * i * ExpertAgent.KART_LENGTH
            # Find if we crash with any kart, as long as we haven't found one yet
            for j, kart_velocity in enumerate(state['karts_velocity']):
                # Ignore karts ahead that are faster than this kart.
                if speed < np.linalg.norm(kart_velocity):
                    continue
                kart_pos = state['karts_position'][j] + kart_velocity * dt * i
                kart_distance = np.linalg.norm(step_coord - kart_pos)
                if kart_distance < ExpertAgent.KART_LENGTH:
                    return j
        return None

    def __get_speed_for_turn_radius(self, curve_radius: float) -> float:
        """Get the recommended speed for a given turn radius.

        Args:
            curve_radius (float): The radius of the turn.

        Returns:
            float: The recommended speed for the turn.
        """
        # Create an interpolation list based on the TURN_RADIUS values
        turn_radius_values = list(sorted(ExpertAgent.TURN_RADIUS.items(), key=lambda x: x[0]))
        interpolation_values = []
        for speed, value in turn_radius_values:
            interpolation_values.append((speed, np.sin(1.0 / value)))
        # Get the recommended speed using linear interpolation
        angle = np.sin(1.0 / curve_radius)

        if len(interpolation_values) == 1:
            return interpolation_values[0][0]

        if interpolation_values[1][1] < interpolation_values[0][1]:  # if decreasing values
            if angle > interpolation_values[0][1]:
                return interpolation_values[0][0]

            last = len(interpolation_values)

            for i in range(1, last):  # for i < last
                if angle < interpolation_values[i][1]:
                    continue
                return interpolation_values[i - 1][0] + (interpolation_values[i][0] - interpolation_values[i - 1][0]) * ((angle - interpolation_values[i - 1][1]) / (interpolation_values[i][1] - interpolation_values[i - 1][1]))
            return interpolation_values[last - 1][0]
        else:   # increasing
            if angle < interpolation_values[0][1]:
                return interpolation_values[0][0]

            last = len(interpolation_values)

            for i in range(1, last):  # for i < last
                if angle > interpolation_values[i][1]:
                    continue
                return interpolation_values[i - 1][0] + (interpolation_values[i][0] - interpolation_values[i - 1][0]) * ((angle - interpolation_values[i - 1][1]) / (interpolation_values[i][1] - interpolation_values[i - 1][1]))
            return interpolation_values[last - 1][0]

    def __handle_braking(self, state: State, action: Action, max_turn_speed: float,
                         min_speed: float, current_track_direction: DirectionType):
        """Handles braking logic based on the current speed and maximum turn speed.

        Args:
            state (State): The current state of the environment.
            max_turn_speed (float): The maximum speed allowed for the current turn.
            min_speed (float): The minimum speed threshold to consider braking.
            current_track_direction (DirectionType): The current direction of the track.
        """
        # Set default to no braking
        action['brake'] = False
        # Get current speed
        speed = np.linalg.norm(state['velocity'])
        # If the current track direction is undefined and speed is above min speed, brake
        if (current_track_direction == DirectionType.DIR_UNDEFINED) and (speed > min_speed):
            action['brake'] = True
            return
        # If the current track direction is right or left, speed is above max turn speed and kart is steering too much, brake
        if (current_track_direction == DirectionType.DIR_RIGHT) or (current_track_direction == DirectionType.DIR_LEFT):
            steer_value = state['prev_actions'][-1]['steer']
            if (speed > min_speed) and (speed > max_turn_speed) and (abs(steer_value) > 0.95):
                action['brake'] = True
            return

    def __computeSkill(self, type: SkillType):
        """Computes the skill level for a given skill type based on AI properties and kart boost status.

        Args:
            type (SkillType): The type of skill to compute.

        Returns:
            int: The computed skill level.
        """
        if type == SkillType.ITEM_SKILL:
            item_skill = 0

            if self.config['item_skill'] > 0:
                if self.config['item_skill'] > 5:
                    item_skill = 5
                else:
                    item_skill = self.config['item_skill']
            return item_skill
        elif type == SkillType.NITRO_SKILL:
            nitro_skill = 0

            if self.config['nitro_usage'] > 0:
                if self.config['nitro_usage'] > 4:
                    nitro_skill = 4
                else:
                    nitro_skill = self.config['nitro_usage']
            return nitro_skill

        return 0

    def __estimate_finish_time(self, state: State):
        """Estimates the time remaining to finish the race based on the current state."""
        num_karts = self.env.num_kart
        track_length = self.env.track.length
        full_distance = self.env.config.laps * track_length
        if full_distance == 0:
            full_distance = 1.0   # For 0 lap races avoid warning below
        # Avoid potential problems (floating point issues, coding bug?) if a
        # kart has driven more than the full distance, but not finished:
        # Return the current time plus initial position to spread arrival
        # times a bit. This code should generally not be used at all, it's
        # just here to avoid invalid finishing times.
        kart_overall_distance = state['distance_down_track'] + state['current_lap'] * track_length
        if kart_overall_distance > full_distance:
            return state['game_time']
        # Finish time is the time needed for the whole race with
        # the computed average speed computed. The distance is always positive
        # due to the way m_distance_increase was computed, so average speed
        # is also always positive.
        average_speed = 1.0 if state['game_time'] == 0 else (kart_overall_distance / state['game_time'])
        # Avoid NAN or invalid results when average_speed is very low
        # or negative (which can happen if a kart drives backwards and
        # m_overall distance becomes smaller than -m_distance_increase).
        # In this case set the time to 59 minutes, offset by kart
        # position (to spread arrival times for all karts that arrive
        # even later). This works for up to 60 karts (otherwise the
        # time displayed would overflow to 00:yy).
        if average_speed < 0.01:
            return 59 * 60.0 + ((1.0 - state['rank']) * (num_karts - 1) + 1)

        est_time = state['game_time'] + (full_distance - kart_overall_distance) / average_speed

        # Avoid times > 59:00 - in this case use kart position to spread
        # arrival time so that each kart has a unique value. The pre-condition
        # guarantees that this works correctly (higher position -> less distance
        # covered -> later arrival time).
        if est_time > 59 * 60.0:
            return 59 * 60.0 + ((1.0 - state['rank']) * (num_karts - 1) + 1)

        return est_time

    def __handle_nitro_and_zipper(self, state: State, action: Action, max_safe_speed: float, kart_ahead_id: Optional[int]):
        """Handles nitro and zipper logic based on the current state.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
        """
        # Get time until fadeout
        time_until_fadeout = state['nitro_duration_time_left'] + state['nitro_fadeout_time_left']
        # Compute skills
        # Nitro skill 0 : don't use
        # Nitro skill 1 : don't use if the kart is braking, is not on the ground, has finished the race, has no nitro,
        #                 has a parachute or an anvil attached, or has a plunger in the face.
        #                 Otherwise, use it immediately
        # Nitro skill 2 : Don't use nitro if there is more than 35% of main effect left..
        #                 Use it when at max speed or under 5 of speed (after rescue, etc.). Use it to pass bombs.
        #                 Tries to builds a reserve of 4 energy to use towards the end
        # Nitro skill 3 : Same as level 2, but don't use until 10% of main effect left, and don't use close
        #                 to bad items, and has a target reserve of 8 energy
        # Nitro skill 4 : Same as level 3, but don't use until 50% of fadeout left and ignore the plunger
        #                 and has a target reserve of 12 energy
        nitro_skill = self.__computeSkill(SkillType.NITRO_SKILL)
        item_skill = self.__computeSkill(SkillType.ITEM_SKILL)
        # Set nitro to false by default
        action['nitro'] = False
        # Set energy reserve
        energy_reserve = 0
        if nitro_skill == 2:
            energy_reserve = 4
        elif nitro_skill == 3:
            energy_reserve = 8
        elif nitro_skill == 4:
            energy_reserve = 12
        # Don't use nitro or zipper if we are braking
        if action['brake']:
            return
        # Don't use nitro or zipper if the kart is not on ground
        if state['jumping']:
            return
        # Don't use nitro or zipper when the AI has a plunger in the face!
        # NOTE: This condition didn't get implemented because we don't need to handicap the expert agent with plungers

        # Don't use nitro or zipper if it would make the kart go too fast
        speed = np.linalg.norm(state['velocity'])
        if speed + ExpertAgent.NITRO_MAX_SPEED_INCREASE > max_safe_speed:
            nitro_skill = 0
        # FIXME : as the zipper can give +15, but only gives +5 instant, this may be too conservative
        if speed + ExpertAgent.ZIPPER_MAX_SPEED_INCREASE > max_safe_speed:
            item_skill = 0
        # If a parachute or anvil is attached, the nitro and zipper don't give much
        # benefit. Better wait till later.
        has_slowdown_attachment = state['attachment'] == Attachment.PARACHUTE or state['attachment'] == Attachment.ANVIL
        if has_slowdown_attachment:
            return
        # Don't compute nitro usage if we don't have nitro
        if state['energy'] == 0:
            nitro_skill = 0
        # If the AI has a lot of nitro, it should consume it without waiting for some fadeout
        big_reserve = False
        # If this kart is the last kart, set nitro reserve to be at most 2
        if (nitro_skill >= 2) and (state['rank'] == 0.0):
            energy_reserve = 2
        # Estimate time towards the end of the race.
        # Decreases the reserve size when there is an estimate of time remaining
        # to the end of less than 2,5 times the maximum nitro effect duration.
        # This vary depending on kart characteristic.
        # There is a margin because the kart will go a bit faster than predicted
        # by the estimate, because the kart may collect more nitro and because
        # there may be moments when it's not useful to use nitro (parachutes, etc).
        if nitro_skill >= 2 and energy_reserve > 0.0:
            finish = self.__estimate_finish_time(state) - state['game_time']
            max_time_effect = (ExpertAgent.NITRO_DURATION + ExpertAgent.NITRO_FADE_OUT_TIME) / ExpertAgent.NITRO_CONSUMPTION * state['energy'] * 2  # the minimum burst consumes around 0.5 energy
            # The burster forces the AI to consume its reserve by series of 2 bursts
            # Otherwise the bursting differences of the various nitro skill wouldn't matter here
            # In short races, most AI nitro usage may be at the end with the reserve
            # NOTE: Here The burster condition was removed
            if time_until_fadeout >= 0:
                energy_reserve = 0

            if (2.5 * max_time_effect) >= finish:
                # Absolute reduction to avoid small amount of unburned nitro at the end
                energy_reserve = min(energy_reserve, finish / (2.5 * max_time_effect / state['energy']) - 0.5)
            if ((1.8 * max_time_effect) >= finish) or (state['energy'] >= 16):
                big_reserve = True
        # Don't use nitro if there is already a nitro boost active
        # Nitro effect and fadeout may vary between karts type
        # So vary time according to kart properties
        if (((nitro_skill == 4) and (time_until_fadeout >= (ExpertAgent.NITRO_FADE_OUT_TIME * 0.5)) and (not big_reserve))
            or ((nitro_skill == 4) and (time_until_fadeout >= ExpertAgent.NITRO_FADE_OUT_TIME))
            or ((nitro_skill == 3) and (time_until_fadeout >= (ExpertAgent.NITRO_FADE_OUT_TIME + ExpertAgent.NITRO_DURATION * 0.1)))
                or ((nitro_skill == 2) and (time_until_fadeout >= (ExpertAgent.NITRO_FADE_OUT_TIME + ExpertAgent.NITRO_DURATION * 0.35)))):
            nitro_skill = 0
        # If trying to pass a bomb to a kart faster or far ahead, use nitro reserve
        speed = np.linalg.norm(state['velocity'])
        if (state['attachment'] == Attachment.BOMB) and (nitro_skill >= 2) and (energy_reserve > 0):
            if ((kart_ahead_id is not None) and (state['karts_distance'][kart_ahead_id] > 10.0)) or (np.linalg.norm(state['karts_velocity'][kart_ahead_id]) > speed):
                energy_reserve = 0

        # TODO : if a kart behind and reasonably close goes faster
        #        and it has a swatter, use nitro to try and dodge the swatter.

        # Don't use nitro if building an energy reserve
        if state['energy'] <= energy_reserve:
            nitro_skill = 0

        # If basic AI, or if the kart is very slow (e.g. after rescue) but not too much (accident)
        # or if kart is near the base max speed, use nitro.
        # This last condition is to profit from the max speed increase
        # And because it means there should be no slowing down from, e.g. plungers
        # If the kart has a huge reserve and may not have enough time until the end to ue it,
        # relax the speed condition.
        # NOTE: burster line was removed here too
        if nitro_skill == 1 or (nitro_skill > 1 and ((speed < 5 and speed > 2) or (speed > 0.96 * self.max_speed) or (speed > 0.3 * self.max_speed and big_reserve))):
            action['nitro'] = True
        # TODO : for now we don't disable nitro use when close to bananas and gums,
        #        because it hurts more often than not (when the bad item is avoided)
        # NOTE: The following code was commented out to simplify as it doesn't do much
        # if avoid_item_close:
        #     return

        # Use zipper
        if (state['powerup'] == Powerup.ZIPPER) and (item_skill >= 2) and (speed > 1.0) and (state['zipper_duration_time_left'] <= 0):
            path_direction, last_path_id = self.__determine_direction(state, current_path_id=0)
            if path_direction == DirectionType.DIR_STRAIGHT:
                diff = state['paths_distance'][last_path_id][1] - state['paths_distance'][0][0]
                if diff < 0:
                    diff += self.env.track.length
                if diff > self.config['straight_length_for_zipper']:
                    action['fire'] = True

    def __handle_acceleration_and_braking(self, state: State, action: Action, curve_radius: float,
                                          current_track_direction: DirectionType, kart_ahead_id: Optional[int]):
        """Handles acceleration and braking logic based on the current state and action.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
        """
        # Step 1 : determine the appropriate max speed for the curve we are in
        #         (this is also calculated in straights, as there is always a
        #          curve lurking at its end)
        max_turn_speed = self.__get_speed_for_turn_radius(curve_radius) * 1.5
        # A kart will not brake when the speed is already slower than this
        # value. This prevents a kart from going too slow (or even backwards)
        # in tight curves.
        MIN_SPEED = 5.0
        # Step 2 : handle braking (there are some cases who need braking besides
        #          a too great speed, like overtaking the leader in FTL)
        self.__handle_braking(state, action, max_turn_speed, MIN_SPEED, current_track_direction)
        if action['brake']:
            action['acceleration'] = 0.0
            return
        # Step 3 : handle nitro and zipper
        # If a bomb is attached, nitro might already be set.
        # FIXME : the bomb situation should be merged here
        if not action['nitro']:
            self.__handle_nitro_and_zipper(state, action, max_turn_speed, kart_ahead_id)
        # Step 4 : handle plunger effect
        # NOTE: This part was removed as we don't need to handicap the expert agent with plungers
        # Step 5 : finally set acceleration to max (acceleration below max is only useful to handicap the kart)
        action['acceleration'] = 1.0

    def __steer_to_point(self, state: State, point: np.ndarray):
        # The point the kart is aiming at can be reached 'incorrectly' if the
        # point is below the y=x line: Instead of aiming at that point directly
        # the point will be reached on its way 'back' after a more than 90
        # degree turn in the circle, i.e.:
        # |                 So the point p (below the y=x line) can not be
        # |  ---\           reached on any circle directly, so it is reached
        # | /    \          on the indicated way. Since this is not the way
        # |/      p         we expect a kart to drive (it will result in the
        # +--------------   kart doing slaloms, not driving straight), the
        # kart will trigger skidding to allow for sharper turns, and hopefully
        # the situation will change so that the point p can then be reached
        # with a normal turn (it usually works out this way quite easily).
        if abs(point[0]) > abs(point[2]):
            # Explicitely set the steering angle high enough to that the
            # steer function will request skidding. 0.1 is added in case
            # of floating point errors.
            if point[0] > 0:
                return state['max_steer_angle'] * self.config['skidding_threshold'] + 0.1
            else:
                return -state['max_steer_angle'] * self.config['skidding_threshold'] - 0.1
        # Now compute the nexessary radius for the turn. After getting the
        # kart local coordinates for the point to aim at, the kart is at
        # (0,0) facing straight ahead. The center of the rotation is then
        # on the X axis and can be computed by the fact that the distance
        # to the kart and to the point to aim at must be the same:
        # r*r = (r-x)*(r-x) + y*y
        # where r is the radius (= position on the X axis), and x, y are the
        # local coordinates of the point to aim at. Solving for r
        # results in r = (x*x+y*y)/2x
        radius = (point[0] * point[0] + point[2] * point[2]) / (2.0 * point[0])
        # sin(steern_angle) = wheel_base / radius:
        sin_steer_angle = ExpertAgent.WHEEL_BASE / radius
        # If the wheel base is too long (i.e. the minimum radius is too large
        # to actually reach the target), make sure that skidding is used
        if sin_steer_angle <= -1.0:
            return -state['max_steer_angle'] * self.config['skidding_threshold'] - 0.1
        if sin_steer_angle >= 1.0:
            return state['max_steer_angle'] * self.config['skidding_threshold'] + 0.1
        # Calculate the actual steering angle
        steer_angle = np.arcsin(sin_steer_angle)
        # After doing the exact computation, we now return an 'oversteered'
        # value. This actually helps in making tighter turns, and also in
        # very tight turns on narrow roads (where following the circle might
        # actually take the kart off track) it forces smaller turns.
        # It does not actually hurt to steer too much, since the steering
        # will be adjusted every frame.
        return steer_angle * 2.0

    def __can_skid(self, state: State, steer_fraction: float, current_track_direction: DirectionType, last_path_id: Optional[int],
                   curve_center: Optional[np.ndarray], curve_radius: float):
        """Determines if the kart can skid based on the steering fraction.

        Args:
            steer_fraction (float): The fraction of the steering angle.

        Returns:
            bool: True if the kart can skid, False otherwise.
        """
        if abs(steer_fraction) > 1.5:
            # If the kart has to do a sharp turn, but is already skidding, find
            # a good time to release the skid button, since this will turn the
            # kart more sharply:
            if state['prev_actions'][-1]['drift'] and (state['skid_direction'] != 0):
                # If the current turn is not sharp enough, delay releasing
                # the skid button.
                return abs(steer_fraction) < 2.5
            # If the kart is not skidding, now is not a good time to start
            return False

        # No skidding on straights
        if (current_track_direction == DirectionType.DIR_STRAIGHT) or (current_track_direction == DirectionType.DIR_UNDEFINED):
            return False

        assert last_path_id is not None, "last_path_id must be not None if the track direction is neither DIR_STRAIGHT nor DIR_UNDEFINED."
        last_path_position = (from_polar_to_cartesian(state['paths_start'][last_path_id]) + from_polar_to_cartesian(state['paths_end'][last_path_id])) / 2.0
        speed = np.linalg.norm(state['velocity'])

        # Only try skidding when a certain minimum speed is reached.
        if speed < ExpertAgent.SKID_MIN_SPEED:
            return False
        # Estimate how long it takes to finish the curve
        diff_kart = -curve_center
        diff_last = last_path_position - curve_center
        angle = get_xz_angle_between(diff_kart, diff_last)
        length = curve_radius * abs(angle)
        duration = length / speed
        # The estimated skdding time is usually too short - partly because
        # he speed of the kart decreases during the turn, partly because
        # the actual path is adjusted during the turn. So apply an
        # experimentally found factor in to get better estimates.
        duration *= 1.5
        # If the remaining estimated time for skidding is too short, stop
        # it. This code will mostly trigger the bonus at the end of a skid.
        if state['prev_actions'][-1]['drift'] and (state['skid_direction'] != 0) and (duration < 1.0):
            return False
        # Test if the AI is trying to skid against track direction. This
        # can happen if the AI is adjusting steering somewhat (e.g. in a
        # left turn steer right to avoid getting too close to the left
        # border). In this case skidding will be useless.
        elif ((steer_fraction > 0) and (current_track_direction == DirectionType.DIR_LEFT)) or ((steer_fraction < 0) and (current_track_direction == DirectionType.DIR_RIGHT)):
            return False
        # If there is a skidding bonus, try to get it.
        elif (len(ExpertAgent.SKID_BONUS_SPEED) > 0) and (ExpertAgent.SKID_TIME_TILL_BONUS[0] < duration):
            return True
        # Otherwise don't skid
        return False

    def __get_steering_when_skidding(self, state: State, steering: float):
        """Adjusts the steering angle when the kart is skidding.

        Args:
            state (State): The current state of the environment.
            steering (float): The current steering angle.

        Returns:
            float: The adjusted steering angle.
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

    def __set_steering(self, state: State, action: Action, angle: float, dt: float,
                       current_track_direction: DirectionType, last_path_id: Optional[int],
                       curve_center: Optional[np.ndarray], curve_radius: float):
        """Sets the steering action based on the current state.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
            angle (float): The desired steering angle.
            dt (float): The time delta since the last action.
            current_track_direction (DirectionType): The current direction of the track.
            last_path_id (int): The ID of the last path segment.
            curve_center (Optional[np.ndarray]): The center of the curve.
            curve_radius (float): The radius of the curve.
        """
        steer_fraction = angle / state['max_steer_angle'].item()
        # Use a simple finite state machine to make sure to randomly decide
        # whether to skid or not only once per skid section. See docs for
        # m_skid_probability_state for more details.
        # NOTE: Code about the skid probability state was removed for simplicity as we don't need it
        # because we assume the expert agent always skids when needed (i.e., probability of skidding = 1.0)
        action['drift'] = self.enable_drift and self.__can_skid(state, steer_fraction, current_track_direction, last_path_id, curve_center, curve_radius)
        # skid_direction = DirectionType.DIR_RIGHT if steer_fraction > 0 else DirectionType.DIR_LEFT

        # Adjust steer fraction in case to be in [-1,1]
        steer_fraction = max(-1.0, min(1.0, steer_fraction))

        # Restrict steering when a plunger is in the face
        # The degree of restriction depends on item_skill
        # NOTE: This part was removed as we don't need to handicap the expert agent with plungers

        # If we are supposed to skid, but the current steering is still
        # in the wrong direction, don't start to skid just now, since then
        # we can't turn into the direction we want to anymore (see
        # Skidding class)
        if ((state['skid_direction'] == -1) and (steer_fraction > 0.2)) or ((state['skid_direction'] == 1) and (steer_fraction < -0.2)):
            action['drift'] = False
        if action['drift'] and (state['skid_direction'] != 0):
            steer_fraction = self.__get_steering_when_skidding(state, steer_fraction)
            if abs(steer_fraction) > 1.8:
                action['drift'] = False
            steer_fraction = max(-1.0, min(1.0, steer_fraction))

        old_steer = state['prev_actions'][-1]['steer']
        # The AI has its own 'time full steer' value
        max_steer_change = dt / self.config['time_full_steer']
        if old_steer < steer_fraction:
            steer_value = steer_fraction if (old_steer + max_steer_change) > steer_fraction else (old_steer + max_steer_change)
        else:
            steer_value = steer_fraction if (old_steer - max_steer_change) < steer_fraction else (old_steer - max_steer_change)
        # Set the final steer value
        action['steer'] = steer_value

    def __get_distance_to_path_center_at(self, state: State, position: np.ndarray, path_id: int):
        """Calculates the distance from a given position to the center of a path segment.

        Args:
            state (State): The current state of the environment.
            path_id (int): The ID of the path segment.
            position (np.ndarray): The position to calculate the distance from.

        Returns:
            float: The distance to the center of the path segment.
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
        proj_z = proj_length * path_dir
        proj_x = rel_pos - proj_z
        abs_center_path_distance = np.linalg.norm(proj_x)
        sign_center_path_distance = 1.0 if np.dot(proj_x, np.array([path_dir[2], 0.0, -path_dir[0]])) >= 0 else -1.0
        return sign_center_path_distance * abs_center_path_distance

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
        half_path_width = 0.5 * state['paths_width'][path_id][0]
        return (0 <= proj_length) and (proj_length <= path_length) and (proj_width <= half_path_width - 0.5 * ExpertAgent.KART_WIDTH)

    def __find_non_crashing_point(self, state: State):
        """Finds a non-crashing point for the kart to aim at.

        Args:
            state (State): The current state of the environment.

        Returns:
            np.ndarray: The non-crashing point.
        """
        last_path_id = 1
        angle = self.__get_angle_to_next_path(state, current_id=0)
        # The original while(1) loop is replaced with a for loop to avoid
        # infinite loops (which we had once or twice). Usually the number
        # of iterations in the while loop is less than 7.
        # In our case with partial observations, the limit is the number of paths in the state
        state_paths = len(state['paths_start'])
        for _ in range(2, state_paths - 1):
            # target_sector is the sector at the longest distance that we can
            # drive to without crashing with the track.
            target_path_id = last_path_id + 1
            target_path_center = (from_polar_to_cartesian(state['paths_start'][target_path_id]) + from_polar_to_cartesian(state['paths_end'][target_path_id])) / 2.0
            angle1 = self.__get_angle_to_next_path(state, current_id=target_path_id)
            diff = angle1 + angle
            # In very sharp turns this algorithm tends to aim at off track points,
            # resulting in hitting a corner. So test for this special case and
            # prevent a too-far look-ahead in this case
            if abs(diff) > (np.pi / 2.0):
                return target_path_id, target_path_center

            # Direction is a vector from our kart to the sectors we are testing
            direction = target_path_center.copy()
            length = np.linalg.norm(direction)
            steps = int(length / ExpertAgent.KART_LENGTH)
            # That shouldn't happen, but since we had one instance of
            # STK hanging, add an upper limit here (usually it's at most
            # 20 steps)
            steps = max(min(steps, 1000), 3)
            # Protection against having vel_normal with nan values
            if length > 0:
                direction /= length
            # Test if we crash if we drive towards the target sector

            # for i in range(2, steps):
            #     step_coord = direction * i * ExpertAgent.KART_LENGTH
            #     if any([self.__check_if_pos_in_path(state, step_coord, pid) for pid in range(last_path_id + 1)]):
            #         continue
            #     return last_path_id, last_path_center

            # NOTE: Old method
            for i in range(2, steps):
                step_coord = direction * i * ExpertAgent.KART_LENGTH
                # Get distance to center of track at this position
                distance = abs(self.__get_distance_to_path_center_at(state, step_coord, last_path_id))
                # If we are outside, the previous node is what we are looking for
                if distance > 0.5 * state['paths_width'][last_path_id][0] - 0.5 * ExpertAgent.KART_WIDTH:
                    last_path_center = (from_polar_to_cartesian(state['paths_start'][last_path_id]) + from_polar_to_cartesian(state['paths_end'][last_path_id])) / 2.0
                    return last_path_id, last_path_center

            angle = angle1
            last_path_id = target_path_id

        last_path_center = (from_polar_to_cartesian(state['paths_start'][last_path_id]) + from_polar_to_cartesian(state['paths_end'][last_path_id])) / 2.0
        return last_path_id, last_path_center

    def __evaluate_items(self, state: State, item_id: int, current_path_id: int, kart_aim_direction: np.ndarray, items_to_collect: list[tuple[int, int]], items_to_avoid: list[tuple[int, int]]):
        """Evaluates items to collect and avoid based on the kart's aim direction.

        Args:
            state (State): The current state of the environment.
            item_id (int): The ID of the item to evaluate.
            current_path_id (int): The ID of the current path segment.
            kart_aim_direction (np.ndarray): The direction the kart is aiming at.
            items_to_collect (list[tuple[int, int]]): List to store item IDs to collect and their associated path IDs.
            items_to_avoid (list[tuple[int, int]]): List to store item IDs to avoid and their associated path IDs.
        """
        item_position = from_polar_to_cartesian(state['items_position'][item_id])
        item_type = state['items_type'][item_id]
        if (item_type != ItemType.BANANA) and (item_type != ItemType.BUBBLEGUM) and (item_type != ItemType.BONUS_BOX) and (item_type != ItemType.NITRO_BIG) and (item_type != ItemType.NITRO_SMALL):
            return

        avoid = False
        # Negative items to avoid
        if item_type == ItemType.BUBBLEGUM:
            avoid = True
        elif item_type == ItemType.BANANA:
            avoid = True
        # Positive items to collect
        elif item_type == ItemType.NITRO_BIG:
            # Only collect nitro, if it can actually be stored.
            if (state['energy'] + ExpertAgent.NITRO_BIG_CONTAINER) > ExpertAgent.NITRO_MAX:
                return
        elif item_type == ItemType.NITRO_SMALL:
            # Only collect nitro, if it can actually be stored.
            if (state['energy'] + ExpertAgent.NITRO_SMALL_CONTAINER) > ExpertAgent.NITRO_MAX:
                return

        # Ignore items to be collected that are out of our way (though all items
        # to avoid are collected).
        if not avoid:
            angle_to_item = get_xz_angle_between(kart_aim_direction, item_position)
            # The kart is driving at high speed, when the current max speed
            # is higher than the max speed of the kart (which is caused by
            # any powerups etc)
            # Otherwise check for skidding. If the kart is still collecting
            # skid bonus, currentMaxSpeed is not affected yet, but it might
            # be if the kart would need to turn sharper, therefore stops
            # skidding, and will get the bonus speed.
            # NOTE: Check if using current speed instead of current max speed is correct
            speed = np.linalg.norm(state['velocity'])
            high_speed = (speed > self.max_speed) or ((state['skid_direction'] != 0) and (state['time_since_skid_start'] >= ExpertAgent.SKID_TIME_TILL_BONUS[0]))
            max_angle = self.config['max_item_angle_high_speed'] if high_speed else self.config['max_item_angle']
            if abs(angle_to_item) > max_angle:
                return
        # Now insert the item into the sorted list of items to avoid
        # (or to collect). The lists are (for now) sorted by distance
        if avoid:
            target_list = items_to_avoid
        else:
            target_list = items_to_collect

        insert_index = 0
        item_distance = np.linalg.norm(item_position) ** 2
        for idx, _ in target_list:
            other_item_position = from_polar_to_cartesian(state['items_position'][idx])
            other_item_distance = np.linalg.norm(other_item_position) ** 2
            if item_distance < other_item_distance:
                break
            insert_index += 1

        target_list.insert(insert_index, (item_id, current_path_id))

    def __handle_bubblegum(self, state: State, action: Action, item_skill: int, items_to_collect: list[tuple[int, int]],
                           items_to_avoid: list[tuple[int, int]], kart_behind_id: Optional[int]):
        """Handle bubblegum depending on the chosen strategy.
        - Level 2 : Use the shield immediately after a wait time
        - Level 3 : Use the shield against flyables except cakes. Use the shield against bad attachments and plunger. Use the bubble gum against an enemy close behind, except if holding a swatter.
        - Level 4 : Level 3, and protect against cakes too, and use before hitting gum/banana
        - Level 5 : Level 4, and use before hitting item box, and let plunger hit (can use the shield after), and use against bomb only when the timer ends

        Args:
            state (State): The current state of the environment.
            item_skill (int): The skill level related to item usage.
            items_to_collect (list[tuple[int, int]]): List of item IDs and their associated path IDs to collect.
            items_to_avoid (list[tuple[int, int]]): List of item IDs and their associated path IDs to avoid.
            kart_behind_id (Optional[int]): The ID of the kart behind, if any.
        """
        # NOTE: Code about detecting projectiles was removed because we don't have that info in the state
        # shield_radius = self.config['shield_incoming_radius']
        attachment = state['attachment']

        if (item_skill == 2) and (state['time_since_last_fire'] > 2.0):
            action['fire'] = True
            return
        # NOTE: Same thing for the code here because it uses projectile info
        # Use shield to remove bad attachments
        if ((attachment == Attachment.BOMB) and (item_skill != 5)) or (attachment == Attachment.PARACHUTE) or (attachment == Attachment.ANVIL):
            action['fire'] = True
            return
        # if it is a bomb, wait : we may pass it to another kart before the timer runs out
        if ((item_skill == 5) and (attachment == Attachment.BOMB)):
            if state['attachment_time_left'] < 2.0:
                action['fire'] = True
                return
        # NOTE: Removed code about plunger because we don't want to be handicapped with plungers
        # Use shield if kart is going to hit a bad item (banana or bubblegum)
        if (item_skill == 4) or (item_skill == 5):
            if (state['attachment'] != Attachment.BUBBLEGUM_SHIELD) and (len(items_to_avoid) > 0):
                d = np.linalg.norm(from_polar_to_cartesian(state['items_position'][items_to_avoid[0][0]])) ** 2
                if ((item_skill == 4) and (d < 1.5)) or ((item_skill == 5) and (d < 0.7)):
                    action['fire'] = True
                    return
        # Use shield if kart is going to hit an item box
        if item_skill == 5:
            if (state['attachment'] != Attachment.BUBBLEGUM_SHIELD) and (len(items_to_collect) > 0):
                d = np.linalg.norm(from_polar_to_cartesian(state['items_position'][items_to_collect[0][0]])) ** 2
                if (state['items_type'][items_to_collect[0][0]] == ItemType.BONUS_BOX) and (d < 0.7):
                    action['fire'] = True
                    return
        # Avoid dropping all bubble gums one after another
        if state['time_since_last_fire'] < 2.0:
            return
        # Use bubblegum if the next kart  behind is 'close' and straight behind
        straight_behind = False
        if kart_behind_id is not None:
            behind_position = from_polar_to_cartesian(state['karts_position'][kart_behind_id])
            abs_angle = np.atan2(abs(behind_position[0]), abs(behind_position[2]))
            if abs_angle < 0.2:
                straight_behind = True

            # NOTE: Some conditions were removed here to simplify the logic
            if (state['karts_distance'][kart_behind_id] < 8.0) and straight_behind:
                action['fire'] = True
                return

    def __handle_cake(self, state: State, action: Action, item_skill: int,
                      kart_ahead_id: Optional[int], kart_behind_id: Optional[int]):
        """Handle cake depending on the chosen strategy
        - Level 2 : Use the cake against any close vulnerable enemy, with priority to those ahead and close,
        check if the enemy is roughly ahead.
        - Level 3 : Level 2 and don't fire on slower karts
        - Level 4 : Level 3 and fire if the kart has a swatter which may hit us
        - Level 5 : Level 4 and don't fire on a shielded kart if we're just behind (gum)

        Args:
            state (State): The current state of the environment.
            item_skill (int): The skill level related to item usage.
            kart_ahead_id (Optional[int]): The ID of the kart ahead, if any.
            kart_behind_id (Optional[int]): The ID of the kart behind, if any.
        """
        # Leave some time between shots
        if state['time_since_last_fire'] < 2.0:
            return
        # Do not fire if the target is too far
        kart_behind_is_close = (state['karts_distance'][kart_behind_id] < 25.0) if kart_behind_id is not None else False
        kart_ahead_is_close = (state['karts_distance'][kart_ahead_id] < 20.0) if kart_ahead_id is not None else False

        kart_ahead_has_gum = False

        straight_ahead = False
        rather_ahead = False  # used to not fire in big curves
        rather_behind = False

        if kart_ahead_id is not None:
            kart_ahead_has_gum = (state['karts_attachment'][kart_ahead_id] == Attachment.BUBBLEGUM_SHIELD)
            kart_ahead_position = from_polar_to_cartesian(state['karts_position'][kart_ahead_id])
            abs_angle = np.atan2(abs(kart_ahead_position[0]), abs(kart_ahead_position[2]))
            if abs_angle < 0.2:
                straight_ahead = True
            if abs_angle < 0.5:
                rather_ahead = True

        if kart_behind_id is not None:
            behind_position = from_polar_to_cartesian(state['karts_position'][kart_behind_id])
            abs_angle = np.atan2(abs(behind_position[0]), abs(behind_position[2]))
            if abs_angle < 0.5:
                rather_behind = True

        # Do not fire if the kart is driving too slow
        speed = np.linalg.norm(state['velocity'])
        kart_behind_is_slow = (kart_behind_id is not None) and (np.linalg.norm(state['karts_velocity'][kart_behind_id]) < speed)
        kart_ahead_is_slow = (kart_ahead_id is not None) and (np.linalg.norm(state['karts_velocity'][kart_ahead_id]) < speed)
        # Establish which of the target is better if any
        fire_ahead = 0.0
        fire_behind = 0.0

        if kart_behind_is_close and rather_behind:
            fire_behind += 25.0 - state['karts_distance'][kart_behind_id]
        else:
            fire_behind -= 100.0

        if kart_ahead_is_close and rather_ahead:
            fire_ahead += 30.0 - state['karts_distance'][kart_ahead_id]
        else:
            fire_ahead -= 100.0

        # Don't fire on an invulnerable kart
        # NOTE: Code was removed here because we don't have that info in the state

        # Don't fire at a kart that is slower than us. Reason is that
        # we can either save the cake for later since we will overtake
        # the kart anyway, or that this might force the kart ahead to
        # use its nitro/zipper (and then we will shoot since then the
        # kart is faster).
        if item_skill >= 3:
            if kart_behind_is_slow:
                fire_behind -= 50.0
            else:
                fire_behind += 25.0
            if kart_ahead_is_slow:
                fire_ahead -= 50.0
            else:
                fire_ahead += 25.0
        # Try to take out a kart which has a swatter in priority
        if item_skill >= 4:
            kart_behind_has_swatter = False
            if kart_behind_id is not None:
                kart_behind_has_swatter = (state['karts_attachment'][kart_behind_id] == Attachment.SWATTER)
            kart_ahead_has_swatter = False
            if kart_ahead_id is not None:
                kart_ahead_has_swatter = (state['karts_attachment'][kart_ahead_id] == Attachment.SWATTER)
            # If it is slower, the swatter is more dangerous
            if kart_ahead_has_swatter:
                if kart_ahead_is_slow:
                    fire_ahead += 75.0
                else:
                    fire_ahead += 15.0
            if kart_behind_has_swatter:
                if not kart_behind_is_slow:
                    fire_behind += 25.0
        # Don't fire on a kart straight ahead with a bubblegum shield
        if (item_skill == 5) and kart_ahead_has_gum and straight_ahead:
            fire_ahead = -100.0

        # NOTE: Because the action look back is not supported in the environment, we won't use the following code
        # we use instead a more simplified logic: fire if only firing ahead is interesting

        # # Compare how interesting each target is to determine if firing backwards or not
        # fire_backwards = False
        # if fire_behind > fire_ahead:
        #     fire_backwards = True

        # fire = (fire_backwards and (fire_behind > 0.0)) or ((not fire_backwards) and (fire_ahead > 0.0))
        # action['fire'] = fire
        # if fire:
        #     action['look_back'] = fire_backwards

        fire = (fire_ahead > 0.0)
        action['fire'] = fire

    def __handle_bowling(self, state: State, action: Action, item_skill: int,
                         kart_ahead_id: Optional[int], kart_behind_id: Optional[int]):
        """Handle the bowling ball depending on the chosen strategy.
        - Level 2 : Use the bowling ball against enemies straight ahead or straight behind, and not invulnerable, with a 5 second delay
        - Level 3 : Only 3 seconds of delay
        - Level 4 : Same as level 3
        - Level 5 : Level 4 and don't fire on a shielded kart if we're just behind (gum)

        Args:
            state (State): The current state of the environment.
            item_skill (int): The skill level related to item usage.
            kart_ahead_id (Optional[int]): The ID of the kart ahead, if any.
            kart_behind_id (Optional[int]): The ID of the kart behind, if any.
        """
        # Leave more time between bowling balls, since they are
        # slower, so it should take longer to hit something which
        # can result in changing our target.
        if (item_skill == 2) and (state['time_since_last_fire'] < 5.0):
            return
        if (item_skill >= 3) and (state['time_since_last_fire'] < 3.0):
            return

        # Consider angle towards karts
        straight_behind = False
        straight_ahead = False
        if kart_behind_id is not None:
            behind_position = from_polar_to_cartesian(state['karts_position'][kart_behind_id])
            abs_angle = np.atan2(abs(behind_position[0]), abs(behind_position[2]))
            if abs_angle < 0.2:
                straight_behind = True
        if kart_ahead_id is not None:
            kart_ahead_position = from_polar_to_cartesian(state['karts_position'][kart_ahead_id])
            abs_angle = np.atan2(abs(kart_ahead_position[0]), abs(kart_ahead_position[2]))
            if abs_angle < 0.2:
                straight_ahead = True
        # Don't fire if the kart we are aiming at is invulnerable.
        # NOTE: Code was removed here because we don't have that info in the state
        # Don't fire on a kart straight ahead with a bubblegum shield
        if item_skill == 5:
            kart_ahead_has_gum = False
            if kart_ahead_id is not None:
                kart_ahead_has_gum = (state['karts_attachment'][kart_ahead_id] == Attachment.BUBBLEGUM_SHIELD)
            if kart_ahead_has_gum and straight_ahead:
                straight_ahead = False
        # don't fire if there is no vulnerable kart in the line of fire
        if (not straight_ahead) and (not straight_behind):
            return
        # Bowling balls are slower, so only fire on closer karts - but when
        # firing backwards, the kart can be further away, since the ball
        # acts a bit like a mine (and the kart is racing towards it, too)

        # NOTE: Because the action look back is not supported in the environment, we won't use the following code
        # we use instead a more simplified logic: fire if only firing ahead is possible (distance < 10)

        # kart_ahead_distance = state['karts_distance'][kart_ahead_id] if kart_ahead_id is not None else float('inf')
        # kart_behind_distance = state['karts_distance'][kart_behind_id] if kart_behind_id is not None else float('inf')
        # fire_backwards = (straight_behind and (not straight_ahead)) or (straight_behind and (kart_behind_distance < kart_ahead_distance))
        # distance = kart_behind_distance if fire_backwards else kart_ahead_distance
        # fire = (fire_backwards and (distance < 30.0)) or ((not fire_backwards) and (distance < 10.0))
        # action['fire'] = fire
        # if fire:
        #     action['look_back'] = fire_backwards

        distance = state['karts_distance'][kart_ahead_id] if kart_ahead_id is not None else float('inf')
        fire = (straight_ahead and (distance < 10.0))
        action['fire'] = fire

    def __handle_switch(self, state: State, action: Action, item_skill: int, items_to_collect: list[tuple[int, int]],
                        items_to_avoid: list[tuple[int, int]]):
        """Handle switch depending on the chosen strategy.
        - Level 2 : Use the switch after a wait time
        - Level 3 : Same as level 2 but don't fire if close to a good item
        - Level 4 : Same as level 3 and fire if very close to a bad item
        - Level 5 : Use if it makes a better item available, or if very close to a bad item. Don't use it if too close of a good item.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
            item_skill (int): The skill level related to item usage.
            items_to_collect (list[tuple[int, int]]): List of item IDs and their associated path IDs to collect.
            items_to_avoid (list[tuple[int, int]]): List of item IDs and their associated path IDs to avoid.
        """
        # It's extremely unlikely two switches are used close one after another
        if item_skill == 2:
            if state['time_since_last_fire'] > 2.0:
                action['fire'] = True
                return
        elif (item_skill == 3) or (item_skill == 4):
            if (item_skill == 4) and (len(items_to_avoid) > 0):
                d = np.linalg.norm(from_polar_to_cartesian(state['items_position'][items_to_avoid[0][0]])) ** 2
                if d < 2.0:
                    action['fire'] = True
                    return
            elif len(items_to_collect) > 0:
                d = np.linalg.norm(from_polar_to_cartesian(state['items_position'][items_to_collect[0][0]])) ** 2
                if d > 10.0:
                    action['fire'] = True
                    return
            elif state['time_since_last_fire'] > 2.0:
                action['fire'] = True
                return
        # TODO : retrieve ranking powerup class and use it to evaluate the best item
        #        available depending of if the switch is used or not
        #       In the mean time big nitro > item box > small nitro
        # TODO : make steering switch-aware so that the kart goes towards a bad item
        #        and use the switch at the last moment
        # It would also be possible but complicated to check if using the switch will
        # cause another kart not far from taking a bad item instead of a good one
        # It should also be possible but complicated to discard items when a good
        # and a bad one are two close from one another
        elif item_skill == 5:
            # First step : identify the best available item
            bad = 0
            good = 0
            # Good will store 1 for nitro, big or small, 2 for item box
            # Big nitro are usually hard to take for the AI
            for item_id, _ in reversed(items_to_collect):
                if state['items_type'][item_id] == ItemType.BONUS_BOX:
                    good = 2
                    break
                elif (state['items_type'][item_id] == ItemType.NITRO_BIG) or (state['items_type'][item_id] == ItemType.NITRO_SMALL):
                    good = 1
            # Bad will store 2 for bananas, 3 for bubble gum
            for item_id, _ in reversed(items_to_avoid):
                if state['items_type'][item_id] == ItemType.BUBBLEGUM:
                    bad = 3
                    break
                elif state['items_type'][item_id] == ItemType.BANANA:
                    bad = 2
            # Second step : make sure a close item don't make the choice pointless
            if len(items_to_avoid) > 0:
                d = np.linalg.norm(from_polar_to_cartesian(state['items_position'][items_to_avoid[0][0]])) ** 2
                # fire if very close to a bad item
                if d < 2.0:
                    action['fire'] = True
                    return
            if len(items_to_collect) > 0:
                d = np.linalg.norm(from_polar_to_cartesian(state['items_position'][items_to_collect[0][0]])) ** 2
                # don't fire if close to a good item
                if d < 5.0:
                    return
            # Third step : Use or don't use to get the best available item
            if bad > good:
                action['fire'] = True
                return

    def __handle_swatter(self, state: State, action: Action, item_skill: int, kart_ahead_id: Optional[int],
                         kart_behind_id: Optional[int]):
        """Handle the swatter depending on the chosen strategy.
        - Level 2 : Use the swatter immediately after a wait time
        - Level 3 : Use the swatter when enemies are close
        - Level 4 : Level 3 and use the swatter to remove bad attachments
        - Level 5 : Level 4 and use against bomb only when the timer ends

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
            item_skill (int): The skill level related to item usage.
            kart_ahead_id (Optional[int]): The ID of the kart ahead, if any.
            kart_behind_id (Optional[int]): The ID of the kart behind, if any.
        """
        attachment = state['attachment']
        if (item_skill == 2) and (state['time_since_last_fire'] > 2.0):
            action['fire'] = True
            return
        # Use swatter to remove bad attachments
        if (item_skill == 4) or (item_skill == 5):
            if ((attachment == Attachment.BOMB) and (item_skill == 4)) or (attachment == Attachment.PARACHUTE) or (attachment == Attachment.ANVIL):
                action['fire'] = True
                return
            # if it is a bomb, wait : we may pass it to another kart before the timer runs out
            if (item_skill == 5) and (attachment == Attachment.BOMB):
                if state['attachment_time_left'] > 3.0:
                    action['fire'] = True
                    return
        # Squared distance for which the swatter works
        d2 = ExpertAgent.SWATTER_DISTANCE
        # Fire if the closest kart ahead or to the back is not already
        # squashed and close enough.
        # FIXME: this can be improved on, since more than one kart might
        #        be hit, and a kart ahead might not be at an angle at
        #        which the glove can be used.
        # NOTE: Here we simplify the condition because we don't have squashed info in the state
        speed = np.linalg.norm(state['velocity'])
        kart_ahead_speed = np.linalg.norm(state['karts_velocity'][kart_ahead_id]) if kart_ahead_id is not None else 0.0
        kart_ahead_distance2 = state['karts_distance'][kart_ahead_id] ** 2 if kart_ahead_id is not None else float('inf')
        kart_behind_distance2 = state['karts_distance'][kart_behind_id] ** 2 if kart_behind_id is not None else float('inf')
        if ((kart_ahead_id is not None) and (kart_ahead_distance2 < d2) and (kart_ahead_speed < speed)
                or ((kart_behind_id is not None) and (kart_behind_distance2 < d2))):
            action['fire'] = True

    def __handle_items(self, state: State, action: Action, target_position: np.ndarray,
                       target_path_id: int, item_skill: int, kart_ahead_id: Optional[int], kart_behind_id: Optional[int]):
        """Handles item usage logic based on the current state and action.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
            dt (float): The time delta since the last action.
            target_position (np.ndarray): The target position to aim at.
            target_path_id (int): The ID of the target path segment.
            item_skill: The skill level related to item usage.
            kart_ahead_id: The ID of the kart ahead, if any.
            kart_behind_id: The ID of the kart behind, if any.
        """
        action['fire'] = False
        if state['powerup'] == Powerup.NOTHING:
            return
        # TODO: Implement this
        # //time since last shot is meant to avoid using the same item
        # //several times in rapid succession ; not to wait to use a useful
        # //collected item
        # if ( m_kart->getPowerup()->getType() != m_kart->getLastUsedPowerup() )
        # {
        # m_time_since_last_shot = 50.0f; //The AI may wait if the value is low, so set a high value
        # }

        # Tactic 0: don't use item
        # -----------------------------------------
        if item_skill == 0:
            return
        # Tactic 1: wait between 5 and 10 seconds, then use item
        # ------------------------------------------------------
        if item_skill == 1:
            random_time = 5.0 + 5.0 * random.random()
            if state['time_since_last_item_use'] > random_time:
                action['fire'] = True
            return
        # Tactics 2 to 5: calculate
        # -----------------------------------------
        min_bubble_time = 2.0
        # Preparing item list for item aware actions
        # Angle to aim_point
        kart_aim_direction = target_position.copy()

        current_path_id = 0
        distance = 0
        items_to_collect = []
        items_to_avoid = []

        # 1) Filter and sort all items close by
        # -------------------------------------
        max_item_lookahead_distance = 20.0
        while distance < max_item_lookahead_distance:
            items_ahead = [i for i, pos in enumerate(state['items_position']) if self.__check_if_pos_in_path(state, pos, current_path_id)]
            for item_id in items_ahead:
                self.__evaluate_items(state, item_id, current_path_id, kart_aim_direction, items_to_collect, items_to_avoid)
            distance += np.linalg.norm(from_polar_to_cartesian(state['paths_start'][current_path_id]) - from_polar_to_cartesian(state['paths_start'][current_path_id + 1]))
            current_path_id += 1
            if current_path_id == target_path_id:
                break
        # items_to_avoid and items_to_collect now contain the closest item information needed after
        # What matters is (a) if the lists are void ; (b) if they are not, what kind of item it is
        if state['powerup'] == Powerup.BUBBLEGUM:
            self.__handle_bubblegum(state, action, item_skill, items_to_collect, items_to_avoid, kart_behind_id)
        elif state['powerup'] == Powerup.CAKE:
            if not ((state['shield_time'] > min_bubble_time) and ExpertAgent.SHIELD_RESTRICT_WEAPONS):
                self.__handle_cake(state, action, item_skill, kart_ahead_id, kart_behind_id)
        elif state['powerup'] == Powerup.BOWLING:
            # if the kart has a shield, do not break it by using a bowling ball.
            if not ((state['shield_time'] > min_bubble_time) and ExpertAgent.SHIELD_RESTRICT_WEAPONS):
                self.__handle_bowling(state, action, item_skill, kart_ahead_id, kart_behind_id)
        elif state['powerup'] == Powerup.PLUNGER:
            # if the kart has a shield, do not break it by using a plunger.
            # Leave more time after a plunger, since it will take some
            # time before a plunger effect becomes obvious.
            if (not ((state['shield_time'] > min_bubble_time) and ExpertAgent.SHIELD_RESTRICT_WEAPONS)) and (state['time_since_last_fire'] >= 5.0):
                # Plungers can be fired backwards and are faster,
                # so allow more distance for shooting.
                kart_ahead_distance = state['karts_distance'][kart_ahead_id] if kart_ahead_id is not None else float('inf')
                # NOTE: Because the action look back is not supported in the environment, we won't use the following code
                # we use instead a more simplified logic: fire if only firing ahead is possible

                # kart_behind_distance = state['karts_distance'][kart_behind_id] if kart_behind_id is not None else float('inf')
                # fire_backwards = (((kart_behind_id is not None) and (kart_ahead_id is not None) and (kart_behind_distance < kart_ahead_distance))
                #                   or (kart_ahead_id is None))
                # distance = kart_behind_distance if fire_backwards else kart_ahead_distance
                # fire = (distance < 30.0) or (state['time_since_last_fire'] > 10.0)
                # action['fire'] = fire
                # if fire:
                #     action['look_back'] = fire_backwards

                distance = kart_ahead_distance
                fire = (distance < 30.0) or (state['time_since_last_fire'] > 10.0)
                action['fire'] = fire
        elif state['powerup'] == Powerup.SWITCH:
            self.__handle_switch(state, action, item_skill, items_to_collect, items_to_avoid)
        elif state['powerup'] == Powerup.PARACHUTE:
            # Wait one second more than a previous parachute
            if state['time_since_last_fire'] > (ExpertAgent.PARACHUTE_DURATION_OTHER + 1.0):
                action['fire'] = True
        elif state['powerup'] == Powerup.SWATTER:
            # if the kart has a shield, do not break it by using a swatter
            if state['shield_time'] <= min_bubble_time:
                self.__handle_swatter(state, action, item_skill, kart_ahead_id, kart_behind_id)
        elif state['powerup'] == Powerup.RUBBERBALL:
            # if the kart has a shield, do not break it by using a swatter.
            if not ((state['shield_time'] > min_bubble_time) and ExpertAgent.SHIELD_RESTRICT_WEAPONS):
                # Perhaps some more sophisticated algorithm might be useful.
                # For now: fire if there is a kart ahead (which means that
                # this kart is certainly not the first kart)
                action['fire'] = (state['rank'] != 1.0)

    def __steer_to_avoid(self, state: State, items_to_avoid: list[tuple[int, int]], target_point: np.ndarray):
        """Steers the kart to avoid items in the items_to_avoid list.

        Args:
            state (State): The current state of the environment.
            items_to_avoid (list[tuple[int, int]]): List of item IDs and their associated path IDs to avoid.
            target_position (np.ndarray): The target position to aim at.
        """
        final_target_point = target_point.copy()
        # First determine the left-most and right-most item
        left_most_distance = float('inf')
        right_most_distance = -float('inf')
        left_most_item = 0
        right_most_item = 0
        left_most_item_path_id = items_to_avoid[left_most_item][1]
        right_most_item_path_id = items_to_avoid[right_most_item][1]
        for item_id, path_id in items_to_avoid:
            pos = from_polar_to_cartesian(state['items_position'][item_id])
            dist = self.__get_distance_to_path_center_at(state, pos, path_id)
            if dist < left_most_distance:
                left_most_distance = dist
                left_most_item = item_id
                left_most_item_path_id = path_id
            if dist > right_most_distance:
                right_most_distance = dist
                right_most_item = item_id
                right_most_item_path_id = path_id
        # Check if we would drive left of the leftmost or right of the
        # rightmost point - if so, nothing to do.
        # NOTE: All the following code is more simplified logic compared to the C++ version
        left_pos = from_polar_to_cartesian(state['items_position'][left_most_item])
        angle_to_left = get_xz_angle_between(target_point, left_pos)

        item_index = -1
        angle_to_item = 0.0
        item_path_id = -1
        if angle_to_left > 0:
            # Left of leftmost point
            item_index = left_most_item
            angle_to_item = angle_to_left
            item_path_id = left_most_item_path_id
        else:
            right_pos = from_polar_to_cartesian(state['items_position'][right_most_item])
            angle_to_right = get_xz_angle_between(target_point, right_pos)
            if angle_to_right < 0:
                # Right of rightmost point
                item_index = right_most_item
                angle_to_item = angle_to_right
                item_path_id = right_most_item_path_id

        if item_index != -1:
            # Even though we are on the side, we must make sure
            # that we don't hit that item
            # If we don't hit the item on the side, no more tests are necessary
            item_pos = from_polar_to_cartesian(state['items_position'][item_index])
            dist_to_item = np.linalg.norm(item_pos)
            if np.tan(abs(angle_to_item)) * dist_to_item > ExpertAgent.MIN_DISTANCE_TO_AVOID_ITEM:
                return False, final_target_point
            # See if we can avoid this item by driving further to the side
            dist_to_target = np.linalg.norm(target_point)
            target_point_proj = (target_point / dist_to_target) * (item_pos / np.cos(abs(angle_to_item)))
            item_to_target_proj = target_point_proj - item_pos
            step_aside_proj = (item_to_target_proj / np.linalg.norm(item_to_target_proj)) * ExpertAgent.MIN_DISTANCE_TO_AVOID_ITEM
            avoid_point = item_pos + step_aside_proj
            if self.__check_if_pos_in_path(state, avoid_point, item_path_id):
                # We can avoid the item by steering further to the side
                final_target_point = avoid_point
                return True, final_target_point
            # If avoid_point is NULL, it means steering more to the sides
            # brings us off track. In this case just try to steer to the
            # other side (e.g. when hitting a left-most item and the kart can't
            # steer further left, steer a bit to the right of the left-most item
            # (without further tests if we might hit anything else).
            avoid_point = item_pos - step_aside_proj
            final_target_point = avoid_point
            return True, final_target_point

        # At this stage there must be at least two items - if there was
        # only a single item, the 'left of left-most' or 'right of right-most'
        # tests above are had been and an appropriate steering point was already
        # determined.

        # Try to identify two items we are driving between (if the kart is not
        # driving between two items, one of the 'left of left-most' etc.
        # tests before applied and this point would not be reached).
        distances = [float('inf'), float('inf')]
        item_ids = [-1, -1]
        angle_to_items = [0.0, 0.0]
        for item_id, path_id in items_to_avoid:
            pos = from_polar_to_cartesian(state['items_position'][item_id])
            angle_to_item = get_xz_angle_between(target_point, pos)
            dist_to_item = np.linalg.norm(pos)
            dist = np.tan(abs(angle_to_item)) * dist_to_item
            dir = 0 if angle_to_item < 0 else 1
            if dist < distances[dir]:
                distances[dir] = dist
                item_ids[dir] = item_id
                angle_to_items[dir] = angle_to_item

        assert item_ids[0] != -1 and item_ids[1] != -1, "Logic error in __steer_to_avoid: could not find two items to avoid."
        assert item_ids[0] != item_ids[1], "Logic error in __steer_to_avoid: found the same item twice."

        # We are driving between item_to_avoid[index[0]] and ...[1].
        # If we don't hit any of them, just keep on driving as normal
        hit_left = (distances[0] < ExpertAgent.MIN_DISTANCE_TO_AVOID_ITEM)
        hit_right = (distances[1] < ExpertAgent.MIN_DISTANCE_TO_AVOID_ITEM)
        if not hit_left and not hit_right:
            return False, final_target_point
        # If we hit the left item, aim at the right avoidance point
        # of the left item. We might still hit the right item ... this might
        # still be better than going too far off track
        if hit_left:
            left_item_pos = from_polar_to_cartesian(state['items_position'][item_ids[0]])
            dist_to_target = np.linalg.norm(target_point)
            target_point_proj = (target_point / dist_to_target) * (left_item_pos / np.cos(abs(angle_to_items[0])))
            item_to_target_proj = target_point_proj - left_item_pos
            step_aside_proj = (item_to_target_proj / np.linalg.norm(item_to_target_proj)) * ExpertAgent.MIN_DISTANCE_TO_AVOID_ITEM
            avoid_point = left_item_pos - step_aside_proj
            final_target_point = avoid_point
            return True, final_target_point

        # Now we must be hitting the right item, so try aiming at the left
        # avoidance point of the right item.
        right_item_pos = from_polar_to_cartesian(state['items_position'][item_ids[1]])
        dist_to_target = np.linalg.norm(target_point)
        target_point_proj = (target_point / dist_to_target) * (right_item_pos / np.cos(abs(angle_to_items[1])))
        item_to_target_proj = target_point_proj - right_item_pos
        step_aside_proj = (item_to_target_proj / np.linalg.norm(item_to_target_proj)) * ExpertAgent.MIN_DISTANCE_TO_AVOID_ITEM
        avoid_point = right_item_pos - step_aside_proj
        final_target_point = avoid_point
        return True, final_target_point

    def __hit_bad_item_when_aim_at(self, state: State, item_id: int, items_to_avoid: list[tuple[int, int]]):
        """Checks if aiming at a specific item would result in hitting any bad items.

        Args:
            state (State): The current state of the environment.
            item_id (int): The ID of the item to aim at.
            items_to_avoid (list[tuple[int, int]]): List of item IDs and their associated path IDs to avoid.

        Returns:
            bool: True if aiming at the item would hit a bad item, False otherwise.
        """
        pos = from_polar_to_cartesian(state['items_position'][item_id])
        for avoid_item_id, path_id in items_to_avoid:
            avoid_pos = from_polar_to_cartesian(state['items_position'][avoid_item_id])
            angle_to_avoid = get_xz_angle_between(pos, avoid_pos)
            dist_to_avoid = np.linalg.norm(avoid_pos)
            if np.tan(abs(angle_to_avoid)) * dist_to_avoid < ExpertAgent.MIN_DISTANCE_TO_AVOID_ITEM:
                return True
        return False

    def __handle_item_collection_and_avoidance(self, state: State, target_point: np.ndarray,
                                               target_path_id: int):
        """Handles item collection and avoidance logic based on the current state and action.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
            target_position (np.ndarray): The target position to aim at.
            target_path_id (int): The ID of the target path segment.

        Returns:
            np.ndarray: The potentially modified target position.
        """
        current_path_id = 0
        distance = 0
        items_to_collect = []
        items_to_avoid = []
        # 1) Filter and sort all items close by
        # -------------------------------------
        max_item_lookahead_distance = 30.0
        while distance < max_item_lookahead_distance:
            items_ahead = [i for (i, pos) in enumerate(state['items_position']) if self.__check_if_pos_in_path(state, pos, current_path_id)]
            for item_id in items_ahead:
                self.__evaluate_items(state, item_id, current_path_id, target_point, items_to_collect, items_to_avoid)
            distance += np.linalg.norm(from_polar_to_cartesian(state['paths_start'][current_path_id]) - from_polar_to_cartesian(state['paths_start'][current_path_id + 1]))
            current_path_id += 1
            if current_path_id == target_path_id:
                break

        # NOTE: The following logic is simplified compared to the C++ version as it makes the state complicated.
        # 2) If the kart is aiming for an item, but (suddenly) detects
        #    some close-by items to avoid (e.g. behind the item, which was too
        #    far away to be considered earlier, or because the item was switched
        #    to a bad item), the kart cancels collecting the item if this could
        #    cause the item-to-avoid to be collected.
        # --------------------------------------------------------------------
        # 3) Steer towards a pre-selected item
        # -------------------------------------

        # 4) Avoid items-to-avoid
        # -----------------------
        if len(items_to_avoid) > 0:
            # If we need to steer to avoid an item, this takes priority,
            # ignore items to collect and return the new aim_point.
            avoid, target_point = self.__steer_to_avoid(state, items_to_avoid, target_point)
            if avoid:
                return target_point

        # NOTE: This step was removed to simplify the logic (always collect items)
        # 5) We are aiming for a new item. If necessary, determine
        # randomly if this item sshould actually be collected.
        # --------------------------------------------------------

        # 6) Try to aim for items-to-collect
        # ----------------------------------
        if len(items_to_collect) > 0:
            # Aim for the closest item to collect
            item_id, path_id = items_to_collect[0]
            # Test if we would hit a bad item when aiming at this good item.
            # If so, don't change the aim. In this case it has already been
            # ensured that we won't hit the bad item (otherwise steerToAvoid
            # would have detected this earlier).
            if not self.__hit_bad_item_when_aim_at(state, item_id, items_to_avoid):
                # If the item is hit (with the current steering), it means
                # it's on a good enough driveline, so make this item a permanent
                # target. Otherwise only try to get closer (till hopefully this
                # item s on our driveline)
                item_pos = from_polar_to_cartesian(state['items_position'][item_id])
                dist_to_item = np.linalg.norm(item_pos)
                angle_to_item = get_xz_angle_between(target_point, item_pos)
                if np.tan(abs(angle_to_item)) * dist_to_item < ExpertAgent.MIN_DISTANCE_TO_AVOID_ITEM:
                    return target_point
                else:
                    # Kart will not hit item, try to get closer to this item
                    # so that it can potentially become a permanent target.
                    if abs(angle_to_item) < 0.3:
                        return item_pos
        return target_point

    def __handle_steering(self, state: State, action: Action, kart_ahead_id: Optional[int],
                          kart_behind_id: Optional[int], crash_kart_id: Optional[int], dt: float,
                          current_track_direction: DirectionType, last_path_id: Optional[int],
                          curve_center: Optional[np.ndarray], curve_radius: float):
        """Handles steering logic based on the current state and action.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
            kart_ahead_id (Optional[int]): The ID of the kart ahead, if any.
            kart_behind_id (Optional[int]): The ID of the kart behind, if any.
            crash_kart_id (Optional[int]): The ID of the kart that will be crashed into, if any.
            dt (float): The time delta since the last action.
            current_track_direction (DirectionType): The current direction of the track.
            last_path_id (int): The ID of the last path segment.
            curve_center (Optional[np.ndarray]): The center of the curve.
            curve_radius (float): The radius of the curve.
        """
        # Special behaviour if we have a bomb attached: try to hit the kart ahead
        # of us.
        speed = np.linalg.norm(state['velocity'])
        steer_angle = 0.0
        if self.config['handle_bomb'] and (state['attachment'] == Attachment.BOMB) and (kart_ahead_id is not None):
            # TODO: add logic to allow an AI kart to pass the bomb to a kart
            #       close behind by slowing/steering slightly
            # If we are close enough and can pass the bomb, try to hit this kart
            if state['karts_distance'][kart_ahead_id] <= 10:  # NOTE: the condition on previous owner was removed for simplicity
                target_position = from_polar_to_cartesian(state['karts_position'][kart_ahead_id])
                target_speed = np.linalg.norm(state['karts_velocity'][kart_ahead_id])
                if target_speed < speed:
                    time_till_hit = state['karts_distance'][kart_ahead_id] / (speed - target_speed)
                    target_position += state['karts_velocity'][kart_ahead_id] * time_till_hit
                steer_angle = self.__steer_to_point(state, target_position)
                self.__set_steering(state, action, steer_angle, dt, current_track_direction, last_path_id, curve_center, curve_radius)
                return
        # Reaction to being outside of the road
        side_dist = state['center_path_distance'][0]
        item_skill = self.__computeSkill(SkillType.ITEM_SKILL)
        if abs(side_dist) > 0.5 * state['paths_width'][0][0] - 0.5 * ExpertAgent.KART_WIDTH:
            next_path_center = (from_polar_to_cartesian(state['paths_start'][1]) + from_polar_to_cartesian(state['paths_end'][1])) / 2.0
            steer_angle = self.__steer_to_point(state, next_path_center)
        else:
            # # If we are going to crash against a kart, avoid it if it doesn't
            # # drives the kart out of the road
            # # TODO : adds item handling to use a cake if available to
            # # open the road
            # # NOTE: The crashing logic was simplified compared to the C++ version (not considering road crashes)
            # kart_dist_to_path_center = None
            # if crash_kart_id is not None:
            #     # -1 = left, 1 = right, 0 = no crash.
            #     dist_to_path_center = state['center_path_distance'][0]
            #     for i in range(len(state['paths_start'])):
            #         if self.__check_if_pos_in_path(state, state['karts_position'][crash_kart_id], i):
            #             kart_dist_to_path_center = self.__get_distance_to_path_center_at(state, from_polar_to_cartesian(state['karts_position'][crash_kart_id]), i)
            #             break
            # if kart_dist_to_path_center is not None:
            #     if dist_to_path_center > kart_dist_to_path_center:
            #         angle = self.__get_angle_to_next_path(state, current_id=1)
            #         next_path_dir = from_polar_to_cartesian(state['paths_end'][1]) - from_polar_to_cartesian(state['paths_start'][1])
            #         angle_to_dir = get_xz_angle_between(next_path_dir, np.array([0.0, 0.0, 1.0]))
            #         steer_angle = angle - angle_to_dir
            #         steer_angle += 0.5 * np.pi
            #         # normalize steer_angle to [-pi, pi]
            #         if steer_angle > np.pi:
            #             steer_angle -= 2 * np.pi
            #     else:
            #         angle = self.__get_angle_to_next_path(state, current_id=1)
            #         next_path_dir = from_polar_to_cartesian(state['paths_end'][1]) - from_polar_to_cartesian(state['paths_start'][1])
            #         angle_to_dir = get_xz_angle_between(next_path_dir, np.array([0.0, 0.0, 1.0]))
            #         steer_angle = angle - angle_to_dir
            #         steer_angle -= 0.5 * np.pi
            #         # normalize steer_angle to [-pi, pi]
            #         if steer_angle < -np.pi:
            #             steer_angle += 2 * np.pi
            # else:
            # Continue from https://github.com/bpiwowar/pystk2/blob/master/src/karts/controller/skidding_ai.cpp#L466
            # The crashing logic was ignored maybe try to implement it later but if needed
            # Implement findNonCrashingPoint function
            # Implement handling items
            target_path_id, target_point = self.__find_non_crashing_point(state)
            # Manage item utilisation
            self.__handle_items(state, action, target_point, target_path_id, item_skill, kart_ahead_id, kart_behind_id)
            # Potentially adjust the point to aim for in order to either
            # aim to collect item, or steer to avoid a bad item.
            # NOTE: Code about plunger was removed to not handicap the expert with plungers
            if self.config['collect_avoid_items']:
                target_point = self.__handle_item_collection_and_avoidance(state, target_point, target_path_id)
            # Steer to the target point
            steer_angle = self.__steer_to_point(state, target_point)

        self.__set_steering(state, action, steer_angle, dt, current_track_direction, last_path_id, curve_center, curve_radius)

    def normalize_nitro_time(unnormalized_nitro_time: float):
        """Normalizes the nitro time for easier handling in AI computations.

        Args:
            unnormalized_nitro_time (float): The unnormalized nitro time.

        Returns:
            float: The normalized nitro time.
        """
        return (max(0.0, unnormalized_nitro_time - ExpertAgent.NITRO_FADE_OUT_TIME) / ExpertAgent.NITRO_DURATION) * 0.5 + (min(ExpertAgent.NITRO_FADE_OUT_TIME, unnormalized_nitro_time) / ExpertAgent.NITRO_FADE_OUT_TIME) * 0.5

    def unnormalize_nitro_time(normalized_nitro_time: float):
        """Unnormalizes the nitro time back to its original scale.

        Args:
            normalized_nitro_time (float): The normalized nitro time.

        Returns:
            float: The unnormalized nitro time.
        """
        return (max(0.0, normalized_nitro_time - 0.5) / 0.5) * ExpertAgent.NITRO_DURATION + (min(0.5, normalized_nitro_time) / 0.5) * ExpertAgent.NITRO_FADE_OUT_TIME

    def normalize_zipper_time(unnormalized_zipper_time: float):
        """Normalizes the zipper time for easier handling in AI computations.

        Args:
            unnormalized_zipper_time (float): The unnormalized zipper time.

        Returns:
            float: The normalized zipper time.
        """
        return (max(0.0, unnormalized_zipper_time - ExpertAgent.ZIPPER_FADE_OUT_TIME) / ExpertAgent.ZIPPER_DURATION) * 0.5 + (min(ExpertAgent.ZIPPER_FADE_OUT_TIME, unnormalized_zipper_time) / ExpertAgent.ZIPPER_FADE_OUT_TIME) * 0.5

    def unnormalize_zipper_time(normalized_zipper_time: float):
        """Unnormalizes the zipper time back to its original scale.

        Args:
            normalized_zipper_time (float): The normalized zipper time.

        Returns:
            float: The unnormalized zipper time.
        """
        return (max(0.0, normalized_zipper_time - 0.5) / 0.5) * ExpertAgent.ZIPPER_DURATION + (min(0.5, normalized_zipper_time) / 0.5) * ExpertAgent.ZIPPER_FADE_OUT_TIME
