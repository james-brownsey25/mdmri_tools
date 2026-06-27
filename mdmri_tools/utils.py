import numpy as np

def electrostatic_repulsion_sphere(n_points, n_iter=500, step=0.01, rng=None):
    """
    Generate approximately uniformly distributed points on S^2 by
    electrostatic repulsion (Coulomb-like).
    
    Parameters
    ----------
    n_points : int
        Number of points (gradient directions).
    n_iter : int
        Number of gradient-descent steps.
    step : float
        Step size (learning rate) for position updates.
    rng : np.random.Generator or None
        Random generator.

    Returns
    -------
    dirs : (n_points, 3) ndarray
        Unit vectors on the sphere.
    """
    rng = np.random.default_rng() if rng is None else rng

    # Initialise points randomly on the sphere
    x = rng.normal(size=(n_points, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)

    for _ in range(n_iter):
        # Pairwise differences x_i - x_j
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # [N, N, 3]
        dist2 = np.sum(diff**2, axis=-1) + np.eye(n_points)  # [N, N], add diag to avoid div by zero
        inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))          # 1/||r||^3

        # Zero self-interaction
        np.fill_diagonal(inv_dist3, 0.0)

        # Force on each point: sum_j (diff_ij / ||diff_ij||^3)
        forces = (diff * inv_dist3[..., np.newaxis]).sum(axis=1)  # [N, 3]

        # Gradient descent step (move opposite to force to reduce energy)
        x -= step * forces

        # Re-project to the unit sphere
        x /= np.linalg.norm(x, axis=1, keepdims=True)

    return x

def add_rician_noise(x: np.ndarray, sigma: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Add Rician noise to a 1D real-valued NumPy array.

    Parameters
    ----------
    x : np.ndarray
        Input 1D real-valued array.
    sigma : float
        Standard deviation of the zero-mean Gaussian noise added independently
        to the real and imaginary channels.
    rng : np.random.Generator | None, optional
        NumPy random number generator for reproducibility. If None, a new
        default generator is used.

    Returns
    -------
    np.ndarray
        Array with Rician noise added.

    Notes
    -----
    For real-valued input $$x$$, the output is computed as:

    $$y = \sqrt{(x + n_r)^2 + n_i^2}$$

    where $$n_r \sim \mathcal{N}(0, \sigma^2)$$ and
    $$n_i \sim \mathcal{N}(0, \sigma^2)$$ are independent.
    """
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("Input x must be a 1D array.")
    if not np.isrealobj(x):
        raise ValueError("Input x must be real-valued.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    if rng is None:
        rng = np.random.default_rng()

    noise_real = rng.normal(loc=0.0, scale=sigma, size=x.shape)
    noise_imag = rng.normal(loc=0.0, scale=sigma, size=x.shape)

    return np.sqrt((x + noise_real)**2 + noise_imag**2)

def add_rician_noise_multi(signal, sigma, n_realizations, rng=None):
    """
    signal: array of shape (N,)
    sigma: scalar
    returns: array of shape (n_realizations, N)
    """
    if rng is None:
        rng = np.random.default_rng()

    signal = np.asarray(signal)[None, :]  # shape (1, N)
    n1 = rng.normal(0.0, sigma, size=(n_realizations, signal.shape[1]))
    n2 = rng.normal(0.0, sigma, size=(n_realizations, signal.shape[1]))
    return np.sqrt((signal + n1)**2 + n2**2)