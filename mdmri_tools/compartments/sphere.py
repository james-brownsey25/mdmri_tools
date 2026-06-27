from .base import Parameter, BaseCompartment
import numpy as np
from .compartment_utils import normalize_shells, normalize_TE, normalize_gradients, calc_bval
from typing import Any

# Constants:
gamma = 2.6752218744 * 1e8
# gamma = 42.577478518*1e6     # [sec]^-1 * [T]^-1
gamma_ms = gamma * 1e-3  # [ms]^-1 *[T]^-1

class Sphere(BaseCompartment):
    """
    Represents water with gaussian diffusion restricted inside a sphere.

    GPD approximation is used.
    Ref. G.J. Stanisz, A. Szafer, G.A. Wright, R.M. Henkelman
            An analytical model of restricted diffusion in bovine optic nerve
    """

    name = 'sphere'
    parameters = (
        Parameter("D", unit='um^2/ms', required=True),
        Parameter("radius", unit='um', required=True),
        Parameter("T2", unit="ms", required=False, default_value=None),    
        )

    def predict(self, **params: Any) -> np.ndarray:
        """
        Expected keys in params:
            - shells (required): (3, N) array
            - gradients (required): (3, N) or (N, 3) array
            - TE (optional): scalar or 1D

        Also includes microstructural params:
            - D
            - Radius
            - T2 (may be None)
        """
        # ---- 1. Required 'shells' argument ----
        if "shells" not in params:
            raise ValueError(
                f"'shells' argument is required in predict() for {self.name}"
            )
        
        if (params['shells'].ndim==1 and len(params['shells']) != 3) or (params['shells'].ndim==2 and params['shells'].shape[0]!=3):
            raise ValueError(
                f"'shells' argument must be of shape (3, N) for a spherical compartment"
            )

        shells_raw = params["shells"]
        shells_arr, n_shells = normalize_shells(shells_raw)
        G_all, Delta_all, delta_all = shells_arr

        if "g_dirs" not in params:
            raise ValueError(
                f"'g_dirs' argument is required in predict() for {self.name}"
            )
        
        gradients_raw = params["g_dirs"]
        grad_arr, n_dirs = normalize_gradients(gradients_raw)

        # ---- 2. TE handling, conditioned on T2 ----
        T2 = self.values.get("T2", None)  # from self.values unless overridden
        TE_raw = params.get("TE", None)

        if T2 is not None and TE_raw is None:
            raise ValueError(
                f"Compartment '{self.name}' requires TE when T2 is specified."
            )

        TE_arr, n_TE = normalize_TE(TE_raw, n_shells)

        # ---- 4. Extract microstructural parameters ----
        # allow override via params, otherwise use instance values
        radius = self.values.get("radius",None)
        D = self.values.get("D",None)

        if radius is None or D is None:
            raise ValueError(
                f"Missing microstructural parameters D and/or radius for compartment '{self.name}'. "
                f"Pass them when instantiating the object."
            )
        
        radius = np.atleast_1d(radius).astype(float)
        D = np.atleast_1d(D).astype(float)

        if np.isscalar(T2) or T2 is None:
            if T2 is None:
                T2_arr = None
                n_T2 = 1
            else:
                T2_arr = np.atleast_1d(float(T2))         # shape (1,)
                n_T2 = 1
        else:
            T2_arr = np.atleast_1d(T2).astype(float)      # shape (n_T2,)
            n_T2 = T2_arr.size

        n_radii = radius.size
        n_Ds = D.size
        
        # ---- 5. Compute the signal shape ----
        # Internal axis order: (n_shells, n_TE, n_dirs)
        # If TE_arr is None: treat n_TE = 1 but you can omit decay or set TE=0

        # Precompute per-shell quantities

        # G_all, Delta_all, delta_all: (n_shells,)
        G_T_per_micron_all = G_all * 1e-3 * 1e-6   # (n_shells,)
        G_T_all = G_T_per_micron_all               # name alias for clarity

        # Reshape to broadcast as (..., n_shells,...)
        G_T_b = G_T_all[None, None, None, :, None, None]  # (1,1,1,1,n_shells,1,1)

        # Precompute GPDsum
        radius_b = radius[:, None, None]            # (n_radii, 1, 1)
        D_b = D[None, :, None]                      # (1, n_Ds, 1)
        delta_b = delta_all[None, None, :]          # (1, 1, n_shells)
        Delta_b = Delta_all[None, None, :]          # (1, 1, n_shells)

        GPDsum = compute_GPDsum_broadcast(radius_b, D_b, delta_b, Delta_b)
        # GPDsum: (n_radii, n_Ds, n_shells)

        # Reshape for final broadcasting:
        GPDsum_b = GPDsum[:, :, None, :, None, None]
        # -> (n_radii, n_Ds, 1, 1, n_shells, 1, 1)

        log_att = -2. * gamma_ms ** 2 * G_T_b ** 2 * GPDsum_b

        # --- 9. T2 decay broadcasting ---

        if T2_arr is None:
            # no T2 decay
            signal = np.exp(log_att)
        else:
            # T2_arr: (n_T2,)
            # TE_arr: (n_TE,)
            # want T2_decay: (n_T2, n_shells, n_TE)
            T2_arr_b = T2_arr[:, None, None]          # (n_T2, 1, 1)
            TE_arr_b = TE_arr[None, None, :]          # (1, 1, n_TE)

            # TE may differ per shell; if normalize_TE already handles this
            # and TE_arr is actually (n_shells, n_TE), just adapt shapes.
            # Assuming TE_arr: (n_TE,) same for all shells:
            T2_decay = np.exp(-TE_arr_b / T2_arr_b)   # (n_T2, 1, n_TE)

            # Broadcast to full shape:
            T2_decay_b = T2_decay[None, None, :, None, :, None]
            # (1,1,1,n_T2,1,n_TE,1)

            signal = np.exp(log_att) * T2_decay_b
        
        signal = np.repeat(signal, n_dirs, axis=-1)

        return np.squeeze(signal)
    
# From Camino source
#  60 first roots from the equation (am*x)j3/2'(am*x)- 1/2 J3/2(am*x)=0
am = np.array([2.08157597781810, 5.94036999057271, 9.20584014293667,
               12.4044450219020, 15.5792364103872, 18.7426455847748,
               21.8996964794928, 25.0528252809930, 28.2033610039524,
               31.3520917265645, 34.4995149213670, 37.6459603230864,
               40.7916552312719, 43.9367614714198, 47.0813974121542,
               50.2256516491831, 53.3695918204908, 56.5132704621986,
               59.6567290035279, 62.8000005565198, 65.9431119046553,
               69.0860849466452, 72.2289377620154, 75.3716854092873,
               78.5143405319308, 81.6569138240367, 84.7994143922025,
               87.9418500396598, 91.0842274914688, 94.2265525745684,
               97.3688303629010, 100.511065295271, 103.653261271734,
               106.795421732944, 109.937549725876, 113.079647958579,
               116.221718846033, 116.221718846033, 119.363764548757,
               122.505787005472, 125.647787960854, 128.789768989223,
               131.931731514843, 135.073676829384, 138.215606107009,
               141.357520417437, 144.499420737305, 147.641307960079,
               150.783182904724, 153.925046323312, 157.066898907715,
               166.492397790874, 169.634212946261, 172.776020008465,
               175.917819411203, 179.059611557741, 182.201396823524,
               185.343175558534, 188.484948089409, 191.626714721361])
    
def compute_GPDsum(am_r, pulse_duration, diffusion_time, diffusivity, radius):
    dam = diffusivity * am_r * am_r
    e11 = -dam * pulse_duration
    e2 = -dam * diffusion_time
    dif = diffusion_time - pulse_duration
    e3 = -dam * dif
    plus = diffusion_time + pulse_duration
    e4 = -dam * plus
    nom = 2 * dam * pulse_duration - 2 + (2 * np.exp(e11)) + (2 * np.exp(e2)) - np.exp(e3) - np.exp(e4)
    denom = dam ** 2 * am_r ** 2 * (radius ** 2 * am_r ** 2 - 2)
    return np.sum(nom / denom)

def compute_GPDsum_broadcast(radius, diffusivity, pulse_duration, diffusion_time):
    """
    radius        : (n_radii, 1, 1)      # in µm
    diffusivity   : (1, n_D, 1)          # in µm^2/ms
    pulse_duration: (1, 1, n_shells)     # delta, in ms
    diffusion_time: (1, 1, n_shells)     # Delta, in ms

    Returns:
        GPDsum: (n_radii, n_D, n_shells)
    """

    # radius: (n_radii, 1, 1)
    # diffusivity: (1, n_D, 1)
    # pulse_duration, diffusion_time: (1, 1, n_shells)
    #
    # Build am_r = am / radius with a root axis at the end.

    # am: (n_roots,) -> (1, 1, 1, n_roots)
    am_vec = am[None, None, None, :]

    # radius: (n_radii, 1, 1) -> (n_radii, 1, 1, 1)
    radius4 = radius[..., None]

    # am_r: (n_radii, 1, 1, n_roots)
    am_r = am_vec / radius4

    # diffusivity: (1, n_D, 1) -> (1, n_D, 1, 1)
    D4 = diffusivity[..., None]

    # pulse_duration, diffusion_time: (1, 1, n_shells) -> (1, 1, n_shells, 1)
    delta4 = pulse_duration[..., None]
    Delta4 = diffusion_time[..., None]

    # dam: (n_radii, n_D, n_shells, n_roots)
    dam = D4 * (am_r**2)

    e11 = -dam * delta4
    e2  = -dam * Delta4
    dif = diffusion_time[..., None] - pulse_duration[..., None]  # or Delta4 - delta4
    e3  = -dam * dif
    plus = Delta4 + delta4
    e4  = -dam * plus

    nom = (
        2.0 * dam * delta4
        - 2.0
        + 2.0 * np.exp(e11)
        + 2.0 * np.exp(e2)
        - np.exp(e3)
        - np.exp(e4)
    )

    # denom: uses am_r and radius exactly as in scalar version
    denom = (
        dam**2
        * am_r**2
        * (radius4**2 * am_r**2 - 2.0)
    )

    GPDsum = np.sum(nom / denom, axis=-1)  # sum over roots
    # shape: (n_radii, n_D, n_shells)

    return GPDsum