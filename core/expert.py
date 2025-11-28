import numpy as np
from enum import Enum
from typing import TypedDict, Optional
from core.state import State, Phase, Attachment, Powerup
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


class SkidProbabilityState(Enum):
    """This implements a simple finite state machine: it starts in
    NOT_YET. The first time the AI decides to skid, the state is changed
    randomly (depending on skid probability) to NO_SKID or SKID.
    As long as the AI keeps on deciding to skid, the state remains
    unchanged (so no new random decision is made) till it decides
    not to skid. In which case the state is set to NOT_YET again.
    This guarantees that for each 'skidable' section of the track
    the random decision is only done once."""
    SKID_PROBAB_NOT_YET = 0
    SKID_PROBAB_NO_SKID = 1
    SKID_PROBAB_SKID = 2


class PointSelectionAlgorithm(Enum):
    """Determines the algorithm to use to select the point-to-aim-for.
    There are two different Point Selection Algorithms:
    1. findNonCrashingPoint() is the default (which is actually slightly buggy,
    but so far best one after handling of 90 degree turns was added).
    2. findNonCrashingPointNew() A newly designed algorithm, which is
    faster than a fixed version of findNonCrashingPoint, but does not
    give as good results as the 'buggy' one.\\
    \\
    So far the default one has by far the best performance, even though it has bugs."""
    PSA_DEFAULT = 0
    PSA_NEW = 1


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
    point_selection_algorithm: PointSelectionAlgorithm
    """The algorithm to use to select the point-to-aim-for"""
    config: ExpertAgentConfig
    """Configuration options for the expert agent (these options control the difficulty of the agent)"""

    # NOTE: These values assume that we always have the same type of kart (the default penguin kart)
    # see https://github.com/bpiwowar/pystk2/blob/f86f21efbf137c95c26f85d113dd1525a8fc4ea6/data/kart_characteristics.xml for more details
    START_DELAYS = [1, 2, 3, 4]
    # START_DELAY_PROBS = [0.43, 0.22, 0.28, 0.07]
    # START_DELAY_BOOSTS = [7.0, 7.0, 4.0, 4.0]
    START_DELAY_PROBS = [0.51, 0.26, 0.19, 0.04]
    TURN_RADIUS = {0: 2.3, 10: 8.625, 25: 17.25, 45: 34.5}
    # nitro
    NITRO_DURATION = 1.0  # seconds
    NITRO_FADE_OUT_TIME = 2.4  # seconds
    NITRO_MAX_SPEED_INCREASE = 5.0  # m/s
    NITRO_CONSUMPTION = 1.25
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
    # general characteristics
    MAX_SPEED = 25.0  # m/s
    WHEEL_BASE = 1.0084210634231567  # meters
    KART_WIDTH = 0.821  # meters
    KART_HEIGHT = 0.675  # meters
    KART_LENGTH = 1.437  # meters

    def __init__(self, env: Env, config: ExpertAgentConfig):
        """Initializes the ExpertAgent with default settings."""
        self.env = env.unwrapped
        self.config = config
        self.point_selection_algorithm = PointSelectionAlgorithm.PSA_DEFAULT

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
        # Determine track direction
        track_direction, curve_center, curve_radius, last_path_id = self.__determine_track_direction(state)
        # Handle acceleration and braking
        self.__handle_acceleration_and_braking(state, action, curve_radius, track_direction,
                                               kart_ahead_id)
        # Handle steering
        self.__handle_steering(state, action, kart_ahead_id, dt=self.env.config.step_size,
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
        # Do nothing if the phase is still not set phase
        if state['phase'].value < Phase.SET_PHASE.value:
            return
        # Sample a start delay that the agent would use
        start_delay = random.choices(ExpertAgent.START_DELAYS, weights=ExpertAgent.START_DELAY_PROBS)[0]
        # If the agent didn't reach the delay yet, do nothing
        if state['nb_steps_since_set'] < start_delay:
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
        time_until_fadeout = state['nitro_time_left']
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
            # NOTE: the following code is commented because there is no opponent kart speed in the state
            # if kart_ahead_position[2] > 10.0 or m_kart_ahead_speed > speed:
            #     energy_reserve = 0
            if ((kart_ahead_id is not None) and (state['karts_position'][kart_ahead_id][2] > 10.0)) or (np.linalg.norm(state['karts_velocity'][kart_ahead_id]) > speed):
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
        if nitro_skill == 1 or (nitro_skill > 1 and ((speed < 5 and speed > 2) or (speed > 0.96 * ExpertAgent.MAX_SPEED) or (speed > 0.3 * ExpertAgent.MAX_SPEED and big_reserve))):
            action['nitro'] = True
        # TODO : for now we don't disable nitro use when close to bananas and gums,
        #        because it hurts more often than not (when the bad item is avoided)
        # NOTE: The following code was commented out to simplify as it doesn't do much
        # if avoid_item_close:
        #     return

        # Use zipper
        if (state['powerup'] == Powerup.ZIPPER) and (item_skill >= 2) and (speed > 1.0) and ((state['zipper_time_left'] - ExpertAgent.ZIPPER_FADE_OUT_TIME) <= 0):
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
        print("Curve center:", curve_center)
        print("Curve radius:", curve_radius)
        print("Angle:", angle)
        print("Length of curve segment:", length)
        print("Estimated skid duration:", duration)
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
        action['drift'] = self.__can_skid(state, steer_fraction, current_track_direction, last_path_id, curve_center, curve_radius)
        print("Current track direction:", current_track_direction)
        print("Speed:", np.linalg.norm(state['velocity']))
        print("Previous skid direction:", state['skid_direction'])
        print(f"Steering angle: {angle}, steer fraction: {steer_fraction}, can skid: {action['drift']}")
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
        print(f"Old steer: {old_steer}, Steer fraction: {steer_fraction}, Max steer change: {max_steer_change}")
        if old_steer < steer_fraction:
            steer_value = steer_fraction if (old_steer + max_steer_change) > steer_fraction else (old_steer + max_steer_change)
        else:
            steer_value = steer_fraction if (old_steer - max_steer_change) < steer_fraction else (old_steer - max_steer_change)
        # Set the final steer value
        print("Final steer value:", steer_value)
        action['steer'] = steer_value

    def __get_distance_to_path_center_at(self, path_start: np.ndarray, path_end: np.ndarray, position: np.ndarray):
        """Calculates the distance from a given position to the center of a path segment.

        Args:
            path_start (np.ndarray): The start point of the path segment.
            path_end (np.ndarray): The end point of the path segment.
            position (np.ndarray): The position to calculate the distance from.

        Returns:
            float: The absolute distance to the center of the path segment.
        """
        rel_path_start = path_start - position
        rel_path_end = path_end - position
        path_dir = rel_path_end - rel_path_start
        x_orth = np.dot(path_dir, rel_path_start) * path_dir / np.linalg.norm(path_dir) ** 2 - rel_path_start
        center_path_distance = np.linalg.norm(x_orth)
        return center_path_distance

    def check_if_pos_in_path(self, state: State, position: np.ndarray, path_id: int):
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
        for _ in range(1, state_paths - 1):
            # target_sector is the sector at the longest distance that we can
            # drive to without crashing with the track.
            last_path_center = (from_polar_to_cartesian(state['paths_start'][last_path_id]) + from_polar_to_cartesian(state['paths_end'][last_path_id])) / 2.0
            angle1 = self.__get_angle_to_next_path(state, current_id=last_path_id)
            diff = angle1 + angle
            # In very sharp turns this algorithm tends to aim at off track points,
            # resulting in hitting a corner. So test for this special case and
            # prevent a too-far look-ahead in this case
            if abs(diff) > np.pi / 2.0:
                return last_path_center

            # Direction is a vector from our kart to the sectors we are testing
            direction = last_path_center.copy()
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
            for i in range(2, steps):
                # step_coord = direction * i * ExpertAgent.KART_LENGTH
                # path_test_id = None
                # for curr_path_id in range(last_path_id + 1):
                #     if self.check_if_pos_in_path(state, step_coord, curr_path_id):
                #         path_test_id = curr_path_id
                #         break
                # # If we are outside, the previous node is what we are looking for
                # if path_test_id is None:
                #     return last_path_center
                step_coord = direction * i * ExpertAgent.KART_LENGTH
                # Get distance to center of track at this position
                distance = self.__get_distance_to_path_center_at(from_polar_to_cartesian(state['paths_start'][last_path_id]),
                                                                 from_polar_to_cartesian(state['paths_end'][last_path_id]),
                                                                 step_coord)
                # If we are outside, the previous node is what we are looking for
                if distance > 0.5 * state['paths_width'][last_path_id] - 0.5 * ExpertAgent.KART_WIDTH:
                    return last_path_center
            angle = angle1
            last_path_id += 1

        last_path_center = (from_polar_to_cartesian(state['paths_start'][last_path_id]) + from_polar_to_cartesian(state['paths_end'][last_path_id])) / 2.0
        return last_path_center

    def __handle_steering(self, state: State, action: Action, kart_ahead_id: Optional[int], dt: float,
                          current_track_direction: DirectionType, last_path_id: Optional[int],
                          curve_center: Optional[np.ndarray], curve_radius: float):
        """Handles steering logic based on the current state and action.

        Args:
            state (State): The current state of the environment.
            action (Action): The action to modify.
            kart_ahead_id (Optional[int]): The ID of the kart ahead, if any.
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
        side_dist = state['center_path_distance']
        item_skill = self.__computeSkill(SkillType.ITEM_SKILL)
        if abs(side_dist) > 0.5 * state['paths_width'][0] - 0.5 * ExpertAgent.KART_WIDTH:
            next_path_center = (from_polar_to_cartesian(state['paths_start'][1]) + from_polar_to_cartesian(state['paths_end'][1])) / 2.0
            steer_angle = self.__steer_to_point(state, next_path_center)
        # TODO: there is an elif condition (crash on other kart logic) here in https://github.com/bpiwowar/pystk2/blob/master/src/karts/controller/skidding_ai.cpp#L430
        # Decide whether to implement or not
        else:
            # Continue from https://github.com/bpiwowar/pystk2/blob/master/src/karts/controller/skidding_ai.cpp#L466
            # The crashing logic was ignored maybe try to implement it later but if needed
            # Implement findNonCrashingPoint function
            # Implement handling items
            target_point = self.__find_non_crashing_point(state)
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
