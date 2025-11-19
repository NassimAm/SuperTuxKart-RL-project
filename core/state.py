from typing import TypedDict
from enum import Enum
import numpy as np
import torch


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
    TRACK_INTRO_PHASE = 0
    SETUP_PHASE = 1
    WAIT_FOR_SERVER_PHASE = 2
    SERVER_READY_PHASE = 3
    READY_PHASE = 4
    SET_PHASE = 5
    GO_PHASE = 6
    MUSIC_PHASE = 7
    RACE_PHASE = 8
    DELAY_FINISH_PHASE = 9
    RESULT_DISPLAY_PHASE = 10
    FINISH_PHASE = 11
    IN_GAME_MENU_PHASE = 12
    UNDEFINED_PHASE = 13

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

    attachment: Attachment
    """Current attachement on the kart."""
    attachment_time_left: float
    """Time left for the current attachment."""
    aux_ticks: float
    """Auxiliary ticks for various purposes."""
    center_path: np.ndarray
    """Center path coordinates."""
    center_path_distance: float
    """Distance to the center path."""
    distance_down_track: float
    """Distance traveled down the track."""
    energy: float
    """Current energy level."""
    front: np.ndarray
    """Front vector coordinates of the kart."""
    items_position: np.ndarray
    """Positions of items on the track."""
    items_type: np.ndarray[ItemType]
    """Types of items on the track."""
    jumping: bool
    """Indicates if the kart is jumping."""
    karts_position: np.ndarray
    """Positions of other karts."""
    max_steer_angle: float
    """Maximum steering angle."""
    paths_distance: np.ndarray
    """Distances from the paths starts and ends (vector of dimension 2)."""
    paths_end: np.ndarray
    """End coordinates of the paths."""
    paths_start: np.ndarray
    """Start coordinates of the paths."""
    paths_width: np.ndarray
    """Widths of the paths."""
    phase: Phase
    """Current phase of the race."""
    powerup: Powerup
    """Current powerup type."""
    shield_time: float
    """Time left for the shield."""
    skeed_factor: float
    """Skeed factor of the kart."""
    velocity: np.ndarray
    """Velocity vector of the kart."""

    CONTINUOUS_KEYS = [
        'attachment_time_left',
        'aux_ticks',
        'center_path',
        'center_path_distance',
        'distance_down_track',
        'energy',
        'front',
        'items_position',
        'karts_position',
        'max_steer_angle',
        'paths_distance',
        'paths_end',
        'paths_start',
        'paths_width',
        'shield_time',
        'skeed_factor',
        'velocity'
    ]

    DISCRETE_KEYS = [
        'attachment',
        'items_type',
        'jumping',
        'phase',
        'powerup'
    ]

    def from_env(flattened_state: dict[str, np.ndarray], state_paths: int = 5, state_items: int = 5,
                 state_karts: int = 5) -> 'State':
        """Create a State from a flattened array representation."""

        continuous_state = flattened_state['continuous']
        attachment_time_left = float(continuous_state[0])
        aux_ticks = float(continuous_state[1])
        center_path = continuous_state[2:5]
        center_path_distance = float(continuous_state[5])
        distance_down_track = float(continuous_state[6])
        energy = float(continuous_state[7])
        front = continuous_state[8:11]
        cur_idx = 11
        next_idx = cur_idx + state_items * 3
        items_position = continuous_state[cur_idx:next_idx].reshape(state_items, 3)
        cur_idx = next_idx
        next_idx = cur_idx + state_karts * 3
        karts_position = continuous_state[cur_idx:next_idx].reshape(state_karts, 3)
        cur_idx = next_idx
        next_idx = cur_idx + 1
        max_steer_angle = float(continuous_state[cur_idx])
        cur_idx = next_idx
        next_idx = cur_idx + state_paths * 2
        paths_distance = continuous_state[cur_idx:next_idx].reshape(state_paths, 2)
        cur_idx = next_idx
        next_idx = cur_idx + state_paths * 3
        paths_end = continuous_state[cur_idx:next_idx].reshape(state_paths, 3)
        cur_idx = next_idx
        next_idx = cur_idx + state_paths * 3
        paths_start = continuous_state[cur_idx:next_idx].reshape(state_paths, 3)
        cur_idx = next_idx
        next_idx = cur_idx + state_paths
        paths_width = continuous_state[cur_idx:next_idx]
        cur_idx = next_idx
        next_idx = cur_idx + 1
        shield_time = float(continuous_state[cur_idx])
        cur_idx = next_idx
        next_idx = cur_idx + 1
        skeed_factor = float(continuous_state[cur_idx])
        cur_idx = next_idx
        next_idx = cur_idx + 3
        velocity = continuous_state[cur_idx:next_idx]

        discrete_state = flattened_state['discrete']
        attachment = Attachment(int(discrete_state[0]))
        cur_idx = 1
        next_idx = cur_idx + state_items
        items_type = np.array([ItemType(e) for e in discrete_state[cur_idx:next_idx]])
        cur_idx = next_idx
        next_idx = cur_idx + 1
        jumping = bool(discrete_state[cur_idx])
        cur_idx = next_idx
        next_idx = cur_idx + 1
        phase = Phase(int(discrete_state[cur_idx]))
        cur_idx = next_idx
        next_idx = cur_idx + 1
        powerup = Powerup(int(discrete_state[cur_idx]))

        return State(
            attachment=attachment,
            attachment_time_left=attachment_time_left,
            aux_ticks=aux_ticks,
            center_path=center_path,
            center_path_distance=center_path_distance,
            distance_down_track=distance_down_track,
            energy=energy,
            front=front,
            items_position=items_position,
            items_type=items_type,
            jumping=jumping,
            karts_position=karts_position,
            max_steer_angle=max_steer_angle,
            paths_distance=paths_distance,
            paths_end=paths_end,
            paths_start=paths_start,
            paths_width=paths_width,
            phase=phase,
            powerup=powerup,
            shield_time=shield_time,
            skeed_factor=skeed_factor,
            velocity=velocity
        )

    def to_env(state: 'State') -> dict[str, np.ndarray]:
        """Convert the state to the environment's expected format."""

        continuous_parts = [
            np.array([state['attachment_time_left']]),
            np.array([state['aux_ticks']]),
            state['center_path'].reshape(-1),
            np.array([state['center_path_distance']]),
            np.array([state['distance_down_track']]),
            np.array([state['energy']]),
            state['front'].reshape(-1),
            state['items_position'].reshape(-1),
            state['karts_position'].reshape(-1),
            np.array([state['max_steer_angle']]),
            state['paths_distance'].reshape(-1),
            state['paths_end'].reshape(-1),
            state['paths_start'].reshape(-1),
            state['paths_width'].reshape(-1),
            np.array([state['shield_time']]),
            np.array([state['skeed_factor']]),
            state['velocity'].reshape(-1),
        ]

        discrete_parts = [
            np.array([state['attachment'].value]),
            np.array([e.value for e in state['items_type']]),
            np.array([int(state['jumping'])]),
            np.array([state['phase'].value]),
            np.array([state['powerup'].value]),
        ]

        return {
            'continuous': np.concatenate(continuous_parts),
            'discrete': np.concatenate(discrete_parts)
        }

    def to_model(state: 'State') -> torch.Tensor:
        """Convert the state to the model's expected format."""

        continuous_parts = [
            torch.tensor([state['attachment_time_left']]),
            torch.tensor([state['aux_ticks']]),
            torch.tensor(state['center_path'].reshape(-1)),
            torch.tensor([state['center_path_distance']]),
            torch.tensor([state['distance_down_track']]),
            torch.tensor([state['energy']]),
            torch.tensor(state['front'].reshape(-1)),
            torch.tensor(state['items_position'].reshape(-1)),
            torch.tensor(state['karts_position'].reshape(-1)),
            torch.tensor([state['max_steer_angle']]),
            torch.tensor(state['paths_distance'].reshape(-1)),
            torch.tensor(state['paths_end'].reshape(-1)),
            torch.tensor(state['paths_start'].reshape(-1)),
            torch.tensor(state['paths_width'].reshape(-1)),
            torch.tensor([state['shield_time']]),
            torch.tensor([state['skeed_factor']]),
            torch.tensor(state['velocity'].reshape(-1)),
        ]

        discrete_parts = [
            torch.tensor(state['attachment'].encode()),
            torch.tensor(np.concatenate([e.encode() for e in state['items_type']])),
            torch.tensor([int(state['jumping'])]),
            torch.tensor(state['phase'].encode()),
            torch.tensor(state['powerup'].encode()),
        ]

        return torch.cat(continuous_parts + discrete_parts)
