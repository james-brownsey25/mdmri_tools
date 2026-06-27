import numpy as np
from scipy import interpolate
from scipy.stats import gamma, norm, truncnorm

def surface_cylinder(radius):
    "The surface normalization function for cylinders."
    return np.pi * radius ** 2

def volume_sphere(radius):
    "The volume normalization function for spheres."
    return (4. / 3.) * np.pi * radius ** 3

def gamma_radii_interpolators(gridsize=50,alpha_range=(0.1, 30.0),beta_range=(1e-3, 2.0)):
    """
    Precompute spline interpolators for the start and end radii of the
    volume-weighted truncated gamma axon-radius distribution.

    Returns
    -------
    start_interpolator : tuple
        Spline representation from scipy.interpolate.bisplrep.
    end_interpolator : tuple
        Spline representation from scipy.interpolate.bisplrep.
    """

    start_grid = np.ones([gridsize, gridsize])
    end_grid = np.ones([gridsize, gridsize])

    alpha_linspace = np.linspace(alpha_range[0], alpha_range[1], gridsize)
    beta_linspace = np.linspace(beta_range[0], beta_range[1], gridsize)

    for i, alpha in enumerate(alpha_linspace):
        for j, beta in enumerate(beta_linspace):
            gamma_distribution = gamma(alpha, scale=beta)
            outer_limit = (
                gamma_distribution.mean() + 9 * gamma_distribution.std())
            x_grid = np.linspace(1e-2, outer_limit, 500)
            pdf = gamma_distribution.pdf(x_grid)
            pdf *= surface_cylinder(x_grid)
            cdf = np.cumsum(pdf)
            cdf /= cdf.max()
            inverse_cdf = np.cumsum(pdf[::-1])[::-1]
            inverse_cdf /= inverse_cdf.max()
            end_grid[i, j] = x_grid[np.argmax(cdf > 0.995)]
            start_grid[i, j] = x_grid[np.argmax(inverse_cdf < 0.995)]
    start_grid = np.clip(start_grid, 1e-2, np.inf)
    end_grid = np.clip(end_grid, 1e-1, np.inf)

    alpha_grid, beta_grid = np.meshgrid(alpha_linspace, beta_linspace)

    start_interpolator = interpolate.bisplrep(alpha_grid.ravel(),
                                                    beta_grid.ravel(),
                                                    start_grid.T.ravel(),
                                                    kx=2, ky=2)

    end_interpolator = interpolate.bisplrep(alpha_grid.ravel(),
                                                    beta_grid.ravel(),
                                                    end_grid.T.ravel(),
                                                    kx=2, ky=2)

    return start_interpolator, end_interpolator


def truncnorm_radii_interpolators(gridsize=50,mu_range=(0.05, 10.0),sigma_range=(0.01, 3.0)):
    """
    Precompute spline interpolators for the start and end radii of the
    volume-weighted truncated Gaussian sphere-radius distribution.

    Returns
    -------
    start_interpolator : tuple
        Spline representation from scipy.interpolate.bisplrep.
    end_interpolator : tuple
        Spline representation from scipy.interpolate.bisplrep.
    """

    start_grid = np.ones((gridsize, gridsize))
    end_grid = np.ones((gridsize, gridsize))

    mu_linspace = np.linspace(mu_range[0], mu_range[1], gridsize)
    sigma_linspace = np.linspace(sigma_range[0], sigma_range[1], gridsize)

    for i, mu in enumerate(mu_linspace):
        for j, sigma in enumerate(sigma_linspace):
            a = -mu / sigma
            b = np.inf
            dist = truncnorm(a=a, b=b, loc=mu, scale=sigma)

            outer_limit = max(mu + 9 * sigma, 1e-2)
            x_grid = np.linspace(1e-8, outer_limit, 500)

            pdf = dist.pdf(x_grid)
            pdf *= volume_sphere(x_grid)

            area = np.trapezoid(pdf, x_grid)
            if area <= 0:
                start_grid[i, j] = 1e-8
                end_grid[i, j] = outer_limit
                continue

            cdf = np.cumsum(pdf)
            cdf /= cdf.max()

            inverse_cdf = np.cumsum(pdf[::-1])[::-1]
            inverse_cdf /= inverse_cdf.max()

            end_grid[i, j] = x_grid[np.argmax(cdf > 0.995)]
            start_grid[i, j] = x_grid[np.argmax(inverse_cdf < 0.995)]

    start_grid = np.clip(start_grid, 1e-8, np.inf)
    end_grid = np.clip(end_grid, 1e-8, np.inf)

    mu_grid, sigma_grid = np.meshgrid(mu_linspace, sigma_linspace)

    start_interpolator = interpolate.bisplrep(
        mu_grid.ravel(),
        sigma_grid.ravel(),
        start_grid.T.ravel(),
        kx=2,
        ky=2
    )

    end_interpolator = interpolate.bisplrep(
        mu_grid.ravel(),
        sigma_grid.ravel(),
        end_grid.T.ravel(),
        kx=2,
        ky=2
    )

    return start_interpolator, end_interpolator

def gamma_radii_pdf(alpha,beta,start_interpolator,end_interpolator,Nsteps=30):

    gamma_dist = gamma(alpha, scale=beta)

    start_point = interpolate.bisplev(alpha, beta, start_interpolator)
    end_point = interpolate.bisplev(alpha, beta, end_interpolator)

    start_point = max(start_point, 1e-8)
    end_point = max(end_point, start_point + 1e-8)

    radii = np.linspace(start_point, end_point, Nsteps)

    normalization = surface_cylinder(radii)
    radii_pdf = gamma_dist.pdf(radii)
    radii_pdf_area = radii_pdf * normalization

    radii_pdf_normalized = (
        radii_pdf_area /
        np.trapezoid(y=radii_pdf_area, x=radii)
    )

    return radii, radii_pdf_normalized

def gaussian_radii_pdf(mu,sigma,start_interpolator,end_interpolator,Nsteps=30):
    """
    Return radii and normalized signal-weighted PDF for a truncated Gaussian
    distribution of sphere radii, weighted by volume.

    Parameters
    ----------
    mu : float
        Mean parameter of the underlying Gaussian before truncation.
    sigma : float
        Standard deviation parameter of the underlying Gaussian before truncation.
    start_interpolator : tuple
        Precomputed bisplrep interpolator for lower radius bound.
    end_interpolator : tuple
        Precomputed bisplrep interpolator for upper radius bound.
    Nsteps : int
        Number of radius samples.

    Returns
    -------
    radii : ndarray
        Radius samples.
    radii_pdf_normalized : ndarray
        Volume-weighted normalized PDF evaluated at radii.
    """

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    a = -mu / sigma
    b = np.inf
    dist = truncnorm(a=a, b=b, loc=mu, scale=sigma)

    start_point = interpolate.bisplev(mu, sigma, start_interpolator)
    end_point = interpolate.bisplev(mu, sigma, end_interpolator)

    start_point = max(start_point, 1e-8)
    end_point = max(end_point, start_point + 1e-8)

    radii = np.linspace(start_point, end_point, Nsteps)

    radii_pdf = dist.pdf(radii)
    radii_pdf_volume = radii_pdf * volume_sphere(radii)

    norm_factor = np.trapezoid(radii_pdf_volume, radii)
    if norm_factor <= 0:
        raise ValueError("Normalization factor is zero or negative")

    radii_pdf_normalized = radii_pdf_volume / norm_factor

    return radii, radii_pdf_normalized