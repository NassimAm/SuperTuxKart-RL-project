from typing import TypedDict
from enum import Enum
import numpy as np
import torch
from core.action import Action

# TODO: check missing attachments, phases, powerups, item types


class Attachment(Enum):
    PARACHUTE = 0
    ANVIL = 1
    BOMB = 2
    SWATTER = 3
    BUBBLEGUM_SHIELD = 6
    NOTHING = 9

    def encode(self):
        """Encode the Attachment value into one-hot encoding."""
        values_list = [e.value for e in Attachment]
        one_hot = np.zeros(len(values_list), dtype=int)
        index = values_list.index(self.value)
        one_hot[index] = 1
        return one_hot

    def decode(one_hot: np.ndarray):
        """Decode a one-hot encoded array into an Attachment value."""
        values_list = [e.value for e in Attachment]
        index = int(np.argmax(one_hot))
        return Attachment(values_list[index])


class Phase(Enum):
    READY_PHASE = 0  # From step 0 to step 10 exclusive
    SET_PHASE = 1  # From step 10 to step 20 exclusive
    RACE_PHASE = 2  # From step 20

    def encode(self):
        """Encode the Phase value into one-hot encoding."""
        values_list = [e.value for e in Phase]
        one_hot = np.zeros(len(values_list), dtype=int)
        index = values_list.index(self.value)
        one_hot[index] = 1
        return one_hot

    def decode(one_hot: np.ndarray):
        """Decode a one-hot encoded array into a Phase value."""
        values_list = [e.value for e in Phase]
        index = int(np.argmax(one_hot))
        return Phase(values_list[index])

    def get_ready_phase_limit() -> int:
        """Get the step limit for the READY_PHASE."""
        return 10

    def get_set_phase_limit() -> int:
        """Get the step limit for the SET_PHASE."""
        return 20


class Powerup(Enum):
    NOTHING = 0
    BUBBLEGUM = 1
    CAKE = 2
    BOWLING = 3
    ZIPPER = 4
    PLUNGER = 5
    SWITCH = 6
    SWATTER = 7
    RUBBERBALL = 8
    PARACHUTE = 9
    ANVIL = 10

    def encode(self):
        """Encode the Powerup value into one-hot encoding."""
        values_list = [e.value for e in Powerup]
        one_hot = np.zeros(len(values_list), dtype=int)
        index = values_list.index(self.value)
        one_hot[index] = 1
        return one_hot

    def decode(one_hot: np.ndarray):
        """Decode a one-hot encoded array into a Powerup value."""
        values_list = [e.value for e in Powerup]
        index = int(np.argmax(one_hot))
        return Powerup(values_list[index])


class ItemType(Enum):
    BONUS_BOX = 0
    BANANA = 1
    NITRO_BIG = 2
    NITRO_SMALL = 3
    BUBBLEGUM = 4
    EASTER_EGG = 6

    def encode(self):
        """Encode the ItemType value into one-hot encoding."""
        values_list = [e.value for e in ItemType]
        one_hot = np.zeros(len(values_list), dtype=int)
        index = values_list.index(self.value)
        one_hot[index] = 1
        return one_hot

    def decode(one_hot: np.ndarray):
        """Decode a one-hot encoded array into an ItemType value."""
        values_list = [e.value for e in ItemType]
        index = int(np.argmax(one_hot))
        return ItemType(values_list[index])


class State(TypedDict):
    """Base class for states in the environment."""

    # Model observations ==========================
    rank: float
    """Normalized position in the race. 0.0 for last place, 1.0 for first place and others in between."""
    attachment: Attachment
    """Current attachement on the kart."""
    attachment_time_left: float
    """Time left for the current attachment."""
    aux_ticks: float  # NOTE: Remove this because it's not useful (always 0.0)
    """Auxiliary ticks for various purposes."""
    center_path: np.ndarray
    """Center path coordinates."""
    center_path_distance: float
    """Distance to the center path."""
    distance_down_track: float  # TODO: normalized?
    """Distance traveled down the track."""
    overall_distance_ratio: float
    """Overall distance ratio in the race."""
    energy: float
    """Current energy level."""
    front: np.ndarray  # NOTE: Removed this because it's not useful (always [0.0, 0.0, 0.7])
    """Front vector coordinates of the kart."""
    items_position: np.ndarray
    """Positions of items on the track."""
    items_type: list[ItemType]
    """Types of items on the track."""
    jumping: bool
    """Indicates if the kart is jumping."""
    stuck: bool
    """Indicates if the kart is stuck."""
    karts_position: np.ndarray  # TODO: at least see one ahead if any and one behind if any
    """Positions of other karts."""
    karts_rel_rank: np.ndarray
    """Ranks of other karts relative to the current kart."""
    karts_velocity: np.ndarray
    """Velocity of other karts."""
    karts_distance: np.ndarray
    """Track distances to other karts."""
    karts_attachment: list[Attachment]
    """Attachments of other karts."""
    max_steer_angle: float
    """Maximum steering angle."""
    paths_distance: np.ndarray  # TODO: normalized? and remove redundant info
    """Distances from the paths starts and ends (vector of dimension 2)."""
    paths_end: np.ndarray
    """End coordinates of the paths."""
    paths_start: np.ndarray
    """Start coordinates of the paths."""
    paths_width: np.ndarray
    """Widths of the paths."""
    phase: Phase
    """Current phase of the race."""
    should_accelerate_in_start: bool
    """Indicates if the kart should accelerate in the start phase."""
    powerup: Powerup
    """Current powerup type."""
    shield_time: float
    """Time left for the shield."""
    skeed_factor: float
    """Skeed factor of the kart."""
    velocity: np.ndarray
    """Velocity vector of the kart."""
    nitro_duration_time_left: float
    """Time left for nitro duration effect."""
    nitro_fadeout_time_left: float
    """Time left for nitro fadeout effect."""
    zipper_duration_time_left: float
    """Time left for zipper duration effect."""
    zipper_fadeout_time_left: float
    """Time left for zipper fadeout effect."""
    skid_direction: int
    """Direction of the skid: -1 for left, 1 for right, 0 for no skid."""
    time_since_last_fire: float
    """Time since the last fire action."""
    prev_actions: list[Action]
    """List of previous actions taken."""
    # Expert observations ===========================
    current_lap: int
    """Current lap number."""
    game_time: float
    """Current game time."""
    time_since_skid_start: float
    """Time since the skid started."""

    CONTINUOUS_KEYS = [
        'rank',
        'attachment_time_left',
        'aux_ticks',
        'center_path',
        'center_path_distance',
        'distance_down_track',
        'overall_distance_ratio',
        'energy',
        'front',
        'items_position',
        'karts_position',
        'karts_rel_rank',
        'karts_velocity',
        'karts_distance',
        'max_steer_angle',
        'paths_distance',
        'paths_end',
        'paths_start',
        'paths_width',
        'nb_steps_since_set',
        'shield_time',
        'skeed_factor',
        'velocity',
        'current_lap',
        'game_time',
        'nitro_time_left',
        'zipper_time_left',
        'skid_direction',
        'time_since_skid_start',
        'time_since_last_fire'
    ]

    DISCRETE_KEYS = [
        'attachment',
        'items_type',
        'jumping',
        'phase',
        'powerup',
        'stuck',
        'karts_attachment',
        'prev_actions'
    ]

    def from_env(state: dict) -> 'State':
        """Create a State from a flattened array representation."""

        return State(
            rank=state['rank'],
            attachment=Attachment(state['attachment']),
            attachment_time_left=state['attachment_time_left'].item(),
            aux_ticks=state['aux_ticks'].item(),
            center_path=state['center_path'],
            center_path_distance=state['center_path_distance'].item(),
            distance_down_track=state['distance_down_track'].item(),
            overall_distance_ratio=state['overall_distance_ratio'],
            energy=state['energy'].item(),
            front=state['front'],
            items_position=state['items_position'],
            items_type=[ItemType(e) for e in state['items_type']],
            jumping=bool(state['jumping']),
            stuck=bool(state['stuck']),
            karts_position=state['karts_position'],
            karts_rel_rank=state['karts_rel_rank'],
            karts_velocity=state['karts_velocity'],
            karts_distance=state['karts_distance'],
            karts_attachment=[Attachment(e) for e in state['karts_attachment']],
            max_steer_angle=state['max_steer_angle'].item(),
            paths_distance=state['paths_distance'],
            paths_end=state['paths_end'],
            paths_start=state['paths_start'],
            paths_width=state['paths_width'].reshape(-1),
            phase=Phase(state['phase']),
            should_accelerate_in_start=bool(state['should_accelerate_in_start']),
            powerup=Powerup(state['powerup']),
            shield_time=state['shield_time'].item(),
            skeed_factor=state['skeed_factor'].item(),
            velocity=state['velocity'],
            current_lap=state['current_lap'],
            game_time=state['game_time'],
            nitro_duration_time_left=state['nitro_duration_time_left'],
            nitro_fadeout_time_left=state['nitro_fadeout_time_left'],
            zipper_duration_time_left=state['zipper_duration_time_left'],
            zipper_fadeout_time_left=state['zipper_fadeout_time_left'],
            skid_direction=state['skid_direction'],
            time_since_skid_start=state['time_since_skid_start'],
            time_since_last_fire=state['time_since_last_fire'],
            prev_actions=[Action.from_env(a) for a in state['prev_actions']]
        )

    def to_env(state: 'State') -> dict[str, np.ndarray]:
        """Convert the state to the environment's expected format."""

        return {
            'rank': state['rank'],
            'attachment': state['attachment'].value,
            'attachment_time_left': state['attachment_time_left'],
            'aux_ticks': state['aux_ticks'],
            'center_path': state['center_path'],
            'center_path_distance': state['center_path_distance'],
            'distance_down_track': state['distance_down_track'],
            'overall_distance_ratio': state['overall_distance_ratio'],
            'energy': state['energy'],
            'front': state['front'],
            'items_position': state['items_position'],
            'items_type': np.array([e.value for e in state['items_type']]),
            'jumping': state['jumping'],
            'stuck': state['stuck'],
            'karts_position': state['karts_position'],
            'karts_rel_rank': state['karts_rel_rank'],
            'karts_velocity': state['karts_velocity'],
            'karts_distance': state['karts_distance'],
            'karts_attachment': np.array([e.value for e in state['karts_attachment']]),
            'max_steer_angle': state['max_steer_angle'],
            'paths_distance': state['paths_distance'],
            'paths_end': state['paths_end'],
            'paths_start': state['paths_start'],
            'paths_width': state['paths_width'],
            'phase': state['phase'].value,
            'should_accelerate_in_start': state['should_accelerate_in_start'],
            'powerup': state['powerup'].value,
            'shield_time': state['shield_time'],
            'skeed_factor': state['skeed_factor'],
            'velocity': state['velocity'],
            'current_lap': state['current_lap'],
            'game_time': state['game_time'],
            'nitro_duration_time_left': state['nitro_duration_time_left'],
            'nitro_fadeout_time_left': state['nitro_fadeout_time_left'],
            'zipper_duration_time_left': state['zipper_duration_time_left'],
            'zipper_fadeout_time_left': state['zipper_fadeout_time_left'],
            'skid_direction': state['skid_direction'],
            'time_since_skid_start': state['time_since_skid_start'],
            'time_since_last_fire': state['time_since_last_fire'],
            'prev_actions': [Action.to_env(a) for a in state['prev_actions']]
        }

    def to_model(state: 'State', previous_action_horizon: int = 1) -> torch.Tensor:
        """Convert the state to the model's expected format."""

        return torch.concat([
            torch.tensor([state['rank']]),
            torch.tensor(state['attachment'].encode()),
            torch.tensor([state['attachment_time_left']]),
            torch.tensor(state['center_path'].reshape(-1)),
            torch.tensor([state['center_path_distance']]),
            torch.tensor([state['distance_down_track']]),
            torch.tensor([state['overall_distance_ratio']]),
            torch.tensor([state['energy']]),
            torch.tensor(state['items_position'].reshape(-1)),
            torch.tensor(np.concatenate([e.encode() for e in state['items_type']])),
            torch.tensor([int(state['jumping'])]),
            torch.tensor([int(state['stuck'])]),
            torch.tensor(state['karts_position'].reshape(-1)),
            torch.tensor(state['karts_rel_rank'].reshape(-1)),
            torch.tensor(state['karts_velocity'].reshape(-1)),
            torch.tensor(state['karts_distance'].reshape(-1)),
            torch.tensor(np.concatenate([e.encode() for e in state['karts_attachment']])),
            torch.tensor([state['max_steer_angle']]),
            torch.tensor(state['paths_distance'].reshape(-1)),
            torch.tensor(state['paths_end'].reshape(-1)),
            torch.tensor(state['paths_start'].reshape(-1)),
            torch.tensor(state['paths_width'].reshape(-1)),
            torch.tensor(state['phase'].encode()),
            torch.tensor([int(state['should_accelerate_in_start'])]),
            torch.tensor(state['powerup'].encode()),
            torch.tensor([state['shield_time']]),
            torch.tensor([state['skeed_factor']]),
            torch.tensor(state['velocity'].reshape(-1)),
            torch.tensor([state['nitro_duration_time_left']]),
            torch.tensor([state['nitro_fadeout_time_left']]),
            torch.tensor([state['zipper_duration_time_left']]),
            torch.tensor([state['zipper_fadeout_time_left']]),
            torch.tensor([state['skid_direction']]),
            torch.tensor([state['time_since_last_fire']]),
            torch.cat([Action.to_model(a) for a in state['prev_actions'][:previous_action_horizon]] if len(state['prev_actions']) > 0 else [Action.to_model(Action.empty()) for _ in range(previous_action_horizon)])
        ])
