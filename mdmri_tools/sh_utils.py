import numpy as np
from dipy.reconst.shm import real_sh_descoteaux

def normalised_shms(bvecs, lmax, normalise_basis=True):
    _, phi, theta = cart2spherical(*bvecs.T)
    y, m, l = real_sh_descoteaux(lmax, theta, phi)
    if normalise_basis:
        y = y/y[0, 0]  # normalisation is required to make the first summary measure represent mean signal
    return y, m, l

def cart2spherical(x, y, z):
    """
    Converts to spherical coordinates

    :param x: x-component of the vector
    :param y: y-component of the vector
    :param z: z-component of the vector
    :return: tuple with (r, phi, theta)-coordinates
    """
    vectors = np.array([x, y, z])
    r = np.sqrt(np.sum(vectors ** 2, 0))
    theta = np.arccos(vectors[2] / r)
    phi = np.arctan2(vectors[1], vectors[0])
    if vectors.ndim == 1:
        if r == 0:
            phi = 0
            theta = 0
    else:
        phi[r == 0] = 0
        theta[r == 0] = 0
    return r, phi, theta

def fit_sh_coeffs(signal, bvecs, sh_degree, normalise_basis):
    """
    Computes coefficients of spherical harmonics for a given diffusion signal.
    :param signal: diffusion signal (n x d) array
    :param bvecs: gradient directions (d x 3) array
    :param sh_degree: maximum degree for spherical harmonics
    :return: coefficients, list of orders
    """

    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    y, m, l = normalised_shms(bvecs, sh_degree, normalise_basis)
    y_inv = np.linalg.pinv(y.T)
    coeffs = signal @ y_inv

    return coeffs, m, l

def fit_reg_sh_coeffs(signal, bvecs, lmax, lb_lambda=0.0, normalise_basis=False):
    """
    Fit real even SH coefficients using regularized least squares.

    Parameters
    ----------
    signal : array, shape (Ndirs,) or (Nsignals, Ndirs)
        Signal values on the sphere.
    bvecs : array, shape (Ndirs, 3)
        Gradient directions.
    lmax : int
        Maximum SH degree.
    lb_lambda : float
        Laplace-Beltrami regularization strength.
    normalise_l0 : bool
        Whether to normalise basis so Y[:,0] corresponds to constant 1.

    Returns
    -------
    coeffs : array, shape (Nsignals, Ncoeff)
    m : array, shape (Ncoeff,)
    l : array, shape (Ncoeff,)
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim == 1:
        signal = signal[None, :]

    Y, m, l = normalised_shms(bvecs, lmax, normalise_basis)

    # Laplace-Beltrami penalty
    lb_eigs = l * (l + 1)
    reg_diag = lb_eigs**2
    reg_diag[l == 0] = 0.0  # never penalize the mean term

    A = Y.T @ Y
    if lb_lambda > 0:
        A = A + lb_lambda * np.diag(reg_diag)

    B = Y.T @ signal.T   # shape (Ncoeff, Nsignals)

    coeffs = np.linalg.solve(A, B).T
    return coeffs, m, l

def calc_rotInvs(coeffs,l_vect,Lmax,normalise=True):

    """
    Return rotational invariants for the SH coefficients
    For each l, we sum the squared m=-l..l coefficients, take the root and multiply by 1/4pi(2l+1).

    coeffs should be of size [N,] or [M,N] where N is the number of sph harm coefficients
    m_vect and l_vect give m and l for each coeff
    """
    if coeffs.ndim==1:
        Sl = np.zeros(int(Lmax/2+1))
        li = np.zeros(int(Lmax/2+1))
        for i,l in enumerate(range(0, Lmax+1, 2)):
            if normalise:
                denominator = 1 / (4 * np.pi * (2 * l + 1))
            else:
                denominator = 1
            Sl[i] = np.sqrt(denominator * np.sum(coeffs[l_vect == l]**2))
            li[i] = l

    elif coeffs.ndim==2:
        Sl = np.zeros((np.shape(coeffs)[0],int(Lmax/2+1)))
        li = np.zeros((np.shape(coeffs)[0],int(Lmax/2+1)))
        for i,l in enumerate(range(0, Lmax+1, 2)):
            if normalise:
                denominator = 1 / (4 * np.pi * (2 * l + 1))
            else:
                denominator = 1
            Sl[:,i] = np.sqrt(denominator * np.sum(coeffs[:,l_vect == l]**2,axis=1))
            li[:,i] = l

            return Sl, li

def get_invariants(signal, G_dirs, lmax):
    coeffs, m_vec, l_vec = fit_sh_coeffs(signal, G_dirs, lmax, normalise_basis=False)
    invars, li = calc_rotInvs(coeffs, l_vec, lmax, normalise=True)
    if coeffs.ndim==1:
        S0 = invars[0]   # spherical mean
        S2 = invars[1]   # spherical variance-like (should be ~0 at b=0)
    elif coeffs.ndim==2:
        S0 = invars[:,0]
        S2 = invars[:,1]
    return S0, S2

def get_reg_invariants(signal, G_dirs, lmax, lb_lambda):
    coeffs, m_vec, l_vec = fit_reg_sh_coeffs(signal, G_dirs, lmax, lb_lambda, normalise_basis=False)
    invars, li = calc_rotInvs(coeffs, l_vec, lmax, normalise=True)
    if coeffs.ndim==1:
        S0 = invars[0]   # spherical mean
        S2 = invars[1]   # spherical variance-like (should be ~0 at b=0)
    elif coeffs.ndim==2:
        S0 = invars[:,0]
        S2 = invars[:,1]
    return S0, S2