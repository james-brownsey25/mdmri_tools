import numpy as np
from typing import Tuple, Optional, Sequence

# Gyromagnetic ratio [rad / (ms * mT)]
gyroMagnRatio = 267.513e-6

def normalize_shells(shells: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Normalizes 'shells' to a canonical internal representation and returns:
        shells_arr, n_shells

    Allowed forms:
        - scalar (0-d or shape=())
        - 1D array of length N
        - 2D array with shape (3, N) where rows represent
          [G, diffusion_time, pulse_duration]

    Disallowed:
        - 2D array with shape (N, 3) (ambiguous: could be 3 b-values)
        - any other shape
    """
    shells_arr = np.asarray(shells)

    # scalar
    if shells_arr.ndim == 0:
        shells_arr = shells_arr.reshape(1)  # treat as length-1 vector
        n_shells = 1
        return shells_arr, n_shells

    # 1D vector of b-values
    if shells_arr.ndim == 1:
        n_shells = shells_arr.shape[0]
        return shells_arr, n_shells

    # 2D - expect shape (3, N)
    if shells_arr.ndim == 2:
        if shells_arr.shape[0] == 3:
            n_shells = shells_arr.shape[1]
            return shells_arr, n_shells
        elif shells_arr.shape[1] == 3:
            raise ValueError(
                "shells with shape (N, 3) is ambiguous. "
                "Use shape (3, N) where rows are [G, diffusion_time, pulse_duration]."
            )
        else:
            raise ValueError(
                f"Invalid shells shape {shells_arr.shape}. "
                "Expected scalar, 1D vector, or (3, N)."
            )

    raise ValueError(
        f"Invalid shells dimension {shells_arr.ndim}. "
        "Expected scalar, 1D, or 2D with shape (3, N)."
    )

def normalize_TE(TE, n_shells: int) -> Tuple[Optional[np.ndarray], int]:
    """
    Normalizes TE to at most 1D array that can broadcast along shells.

    Rules:
        - If TE is None: return (None, 1)  # means no TE dimension
        - If TE is scalar: shape (1,) and n_TE = 1
        - If TE is 1D: shape (N_TE,) and n_TE = N_TE
        - If TE has more than 1 dimension: raise
    """
    if TE is None:
        return None, 1

    TE_arr = np.asarray(TE)

    if TE_arr.ndim == 0:
        TE_arr = TE_arr.reshape(1)  # scalar -> length-1
        n_TE = 1
        return TE_arr, n_TE

    if TE_arr.ndim == 1:
        n_TE = TE_arr.shape[0]
        return TE_arr, n_TE

    raise ValueError(
        f"Invalid TE dimension {TE_arr.ndim}. Expected scalar or 1D vector."
    )

def normalize_gradients(gradients) -> Tuple[Optional[np.ndarray], int]:
    """
    Normalizes gradient directions.

    Allowed forms:
        - None: return (None, 1)  # spherical mean, no direction axis
        - 2D array with shape (N_dir, 3) or (3, N_dir)

    Returns:
        grad_arr, n_dirs

    grad_arr is either None or shape (N_dir, 3).
    """
    if gradients is None:
        return None, 1

    grad_arr = np.asarray(gradients)
    if grad_arr.ndim != 2:
        raise ValueError(
            f"Invalid gradients dimension {grad_arr.ndim}. "
            "Expected 2D array with shape (N_dir, 3) or (3, N_dir)."
        )

    if grad_arr.shape[1] == 3:
        # (N_dir, 3) OK
        n_dirs = grad_arr.shape[0]
        return grad_arr, n_dirs
    elif grad_arr.shape[0] == 3:
        # (3, N_dir) -> transpose
        grad_arr = grad_arr.T
        n_dirs = grad_arr.shape[0]
        return grad_arr, n_dirs

    raise ValueError(
        f"Invalid gradients shape {grad_arr.shape}. "
        "Expected (N_dir, 3) or (3, N_dir)."
    )

def calc_bval(G, pulse_duration, diffusion_time):
    """
    Calculate B given diffusion times and G
    pulse_duration/diffusion_time must be in ms

    Args:
        G: gradient strength in mT/m
        pulse_duration: pulse duration in ms
        diffusion_time: diffusion time in ms
        t_ramp: ramp time in ms

    Returns:
        numpy array or float (Unit ms/mu^2)
    """
    return (G * gyroMagnRatio * pulse_duration)** 2 * (diffusion_time - pulse_duration/3)