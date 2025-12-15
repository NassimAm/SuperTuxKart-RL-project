import numpy as np


def from_polar_to_cartesian(polar_coords: np.ndarray) -> np.ndarray:
    """Convert 3D polar coordinates to Cartesian coordinates.

    Args:
        polar_coords (np.ndarray): An array of shape (..., 3) where each entry is (ZX angle, ZY angle, distance)."""
    zx_angles = polar_coords[..., 0]
    zy_angles = polar_coords[..., 1]
    distances = polar_coords[..., 2]

    x = distances * np.cos(zy_angles) * np.sin(zx_angles)
    y = distances * np.sin(zy_angles)
    z = distances * np.cos(zy_angles) * np.cos(zx_angles)

    return np.stack((x, y, z), axis=-1)


def from_cartesian_to_polar(cartesian_coords: np.ndarray) -> np.ndarray:
    """Convert 3D Cartesian coordinates to polar coordinates.

    Args:
        cartesian_coords (np.ndarray): An array of shape (..., 3) where each entry is (x, y, z)."""
    x = cartesian_coords[..., 0]
    y = cartesian_coords[..., 1]
    z = cartesian_coords[..., 2]

    distances = np.sqrt(x**2 + y**2 + z**2)
    zx_angles = np.arctan2(x, z)
    zy_angles = np.arctan2(y, np.sqrt(x**2 + z**2))

    return np.stack((zx_angles, zy_angles, distances), axis=-1)


def get_xz_angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate the angle in radians between vectors 'v1' and 'v2'"""
    x1, z1 = v1[0], v1[2]
    x2, z2 = v2[0], v2[2]
    angle = -np.arctan2(x1 * z2 - z1 * x2, x1 * x2 + z1 * z2)
    return angle


def rotate(v: np.array, q: np.array):
    """Compute the rotation of vector with a quaternion

    Formula from
    http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/transforms/derivations/vectors/index.htm

    :param v: The 3D vector
    :param q: The 4D quaternion
    :return: The vector after rotation
    """
    x, y, z = v
    qw, qx, qy, qz = q

    return np.array(
        [
            x * (qx * qx + qw * qw - qy * qy - qz * qz)
            + y * (2 * qx * qy - 2 * qw * qz)
            + z * (2 * qx * qz + 2 * qw * qy),
            x * (2 * qw * qz + 2 * qx * qy)
            + y * (qw * qw - qx * qx + qy * qy - qz * qz)
            + z * (-2 * qw * qx + 2 * qy * qz),
            x * (-2 * qw * qy + 2 * qx * qz)
            + y * (2 * qw * qx + 2 * qy * qz)
            + z * (qw * qw - qx * qx - qy * qy + qz * qz),
        ],
        dtype=v.dtype,
    )
