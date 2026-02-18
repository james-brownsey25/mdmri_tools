from .base import Parameter, BaseCompartment
import numpy as np
from .utils import normalize_shells, normalize_TE, normalize_gradients, calc_bval
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

        # ---- 5. Compute the signal shape ----
        # Internal axis order: (n_shells, n_TE, n_dirs)
        # If TE_arr is None: treat n_TE = 1 but you can omit decay or set TE=0

        signal = np.zeros((n_shells, n_TE, n_dirs), dtype=float)

        for i_shell in range(n_shells):
            shell_info = shells_arr[:, i_shell]  # shape (3,)
            G, Delta, delta = shell_info 
            for i_TE in range(n_TE):
                TE_val = TE_arr[i_TE] if TE_arr is not None else None
                for i_dir in range(n_dirs):
                    G_T_per_micron = G * 1e-3 * 1e-6  # [T] * [um]^-1
                    am_r = am[:, np.newaxis] / radius
                    GPDsum = compute_GPDsum(am_r,delta,Delta,D,radius)
                    attenuation = np.exp(-2. * gamma_ms ** 2 * G_T_per_micron ** 2 * GPDsum)
                    if T2 is None:
                        signal[i_shell,i_TE,i_dir] = attenuation
                    else:
                        T2_decay = np.exp(-TE_val/T2)
                        signal[i_shell,i_TE,i_dir] = attenuation * T2_decay

        # remove dimensions of length 1
        return np.squeeze(signal)
    
    def predict_mean(self, **params: Any) -> np.ndarray:
        """
        Expected keys in params:
            - shells (required): (3, N) array
            - TE (optional): scalar or 1D

        Also includes microstructural params:
            - D
            - Radius
            - T2 (may be None)
        """

        # ---- 1. Required 'shells' argument ----
        if "shells" not in params:
            raise ValueError(
                f"'shells' argument is required in predict_mean() for {self.name}"
            )
        shells_raw = params["shells"]
        shells_arr, n_shells = normalize_shells(shells_raw)

        # For restricted compartments, we require (3, N)
        if shells_arr.ndim != 2 or shells_arr.shape[0] != 3:
            raise ValueError(
                f"'shells' must be a 2D array of shape (3, N) with rows [G, diffusion_time, pulse_duration] "
                f"for compartment '{self.name}'. Got shape {shells_arr.shape}."
            )

        if "g_dirs" in params:
            raise ValueError(
                f"'g_dirs' argument only valid when predict() is called for {self.name}"
            )

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
        
        # ---- 5. Compute the signal shape ----
        # Internal axis order: (n_shells, n_TE, n_dirs)
        # If TE_arr is None: treat n_TE = 1 but you can omit decay or set TE=0

        signal = np.zeros((n_shells, n_TE), dtype=float)

        for i_shell in range(n_shells):
            shell_info = shells_arr[:, i_shell]  # shape (3,)
            G, Delta, delta = shell_info 
            for i_TE in range(n_TE):
                TE_val = TE_arr[i_TE] if TE_arr is not None else None
                G_T_per_micron = G * 1e-3 * 1e-6  # [T] * [um]^-1
                am_r = am[:, np.newaxis] / radius
                GPDsum = compute_GPDsum(am_r,delta,Delta,D,radius)
                attenuation = np.exp(-2. * gamma_ms ** 2 * G_T_per_micron ** 2 * GPDsum)
                if T2 is None:
                    signal[i_shell,i_TE] = attenuation
                else:
                    T2_decay = np.exp(-TE_val/T2)
                    signal[i_shell,i_TE] = attenuation * T2_decay

        # remove dimensions of length 1
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