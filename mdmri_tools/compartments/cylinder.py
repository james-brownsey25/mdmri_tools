from .base import Parameter, BaseCompartment
import numpy as np
from .compartment_utils import normalize_shells, normalize_TE, normalize_gradients, calc_bval
from typing import Any
from scipy.integrate import quad

MINIMUM_RADIUS = 1e-10 # to avoid numerical issues
# Constants:
gamma = 2.6752218744 * 1e8
# gamma = 42.577478518*1e6     # [sec]^-1 * [T]^-1
gamma_ms = gamma * 1e-3  # [ms]^-1 *[T]^-1

class Cylinder(BaseCompartment):
    """
    Represents water with gaussian diffusion restricted inside a cylinder.

    Cylinder is oriented in the z-direction.
    """

    name = 'cylinder'
    parameters = (
        Parameter("D", unit='um^2/ms', required=True),
        Parameter("radius", unit='um', required=True),
        Parameter("dir", unit="", required = False, default_value=np.array([0,0,1])),
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
            - Direction
            - T2 (may be None)
        """
        # ---- 1. Required 'shells' argument ----
        if "shells" not in params:
            raise ValueError(
                f"'shells' argument is required in predict() for {self.name}"
            )
        
        if (params['shells'].ndim==1 and len(params['shells']) != 3) or (params['shells'].ndim==2 and params['shells'].shape[0]!=3):
            raise ValueError(
                f"'shells' argument must be of shape (3, N) for a cylindrical compartment"
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
        dirs_raw = self.values.get("dir")

        cyl_dirs_arr, n_cyl_dirs = normalize_gradients(dirs_raw)

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

        # Precompute direction-related terms
        # Normalize cylinder directions (just in case)
        cyl_norm = np.linalg.norm(cyl_dirs_arr, axis=1, keepdims=True) + 1e-12
        cyl_dirs_unit = cyl_dirs_arr / cyl_norm  # (n_cyl_dirs, 3)

        # Dot product between cyl_dirs_unit (k,3) and grad_arr (n_dirs,3)
        # -> (k, n_dirs) via einsum:
        cos_theta = np.einsum("kd,nd->kn", cyl_dirs_unit, grad_arr)
        c2 = cos_theta**2            # (n_cyl_dirs, n_dirs)
        s2 = 1.0 - c2                # (n_cyl_dirs, n_dirs)

        # Broadcast to final shape:
        # We need (1, 1, n_cyl_dirs, 1, 1, 1, n_dirs)
        c2_b = c2[None, None, :, None, None, None, :]     # (1, 1, k, 1, 1, 1, n_dirs)
        s2_b = s2[None, None, :, None, None, None, :]     # same

        # Precompute per-shell quantities

        # G_all, Delta_all, delta_all: (n_shells,)
        G_T_per_micron_all = G_all * 1e-3 * 1e-6   # (n_shells,)
        G_T_all = G_T_per_micron_all               # name alias for clarity

        # b-values: (n_shells,)
        b_all = calc_bval(G_all, delta_all, Delta_all)    # vectorised

        # Reshape to broadcast as (..., n_shells,...)
        G_T_b = G_T_all[None, None, None, None, :, None, None]  # (1,1,1,1,n_shells,1,1)
        b_b = b_all[None, None, None, None, :, None, None]      # same

        # Precompute GPDsum
        radius_b = radius[:, None, None]            # (n_radii, 1, 1)
        D_b = D[None, :, None]                      # (1, n_Ds, 1)
        delta_b = delta_all[None, None, :]          # (1, 1, n_shells)
        Delta_b = Delta_all[None, None, :]          # (1, 1, n_shells)

        GPDsum = compute_GPDsum_broadcast(radius_b, D_b, delta_b, Delta_b)
        # GPDsum: (n_radii, n_Ds, n_shells)

        # Reshape for final broadcasting:
        GPDsum_b = GPDsum[:, :, None, None, :, None, None]
        # -> (n_radii, n_Ds, 1, 1, n_shells, 1, 1)

        # --- 8. Combine terms to get log attenuation ---

        # log_att_perp = -2 * gamma_ms^2 * G_T^2 * GPDsum * s2
        log_att_perp = (
            -2.0 * (gamma_ms**2)
            * (G_T_b**2)
            * GPDsum_b
            * s2_b
        )
        # shape: (n_radii, n_Ds, n_cyl_dirs, 1, n_shells, 1, n_dirs)

        # log_att_para = -b * D * c2
        D_b_full = D[None, :, None, None, None, None, None]
        # -> (1, n_Ds, 1, 1, 1, 1, 1)

        log_att_para = -b_b * D_b_full * c2_b
        # -> (n_radii, n_Ds, n_cyl_dirs, 1, n_shells, 1, n_dirs) via broadcast

        log_att = log_att_perp + log_att_para

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
            T2_decay_b = T2_decay[None, None, None, :, None, :, None]
            # (1,1,1,n_T2,1,n_TE,1)

            signal = np.exp(log_att) * T2_decay_b

        return np.squeeze(signal)
    
# From Camino source
# 60 first roots from the equation j'1(am*x)=0 */
am = np.array([
    1.84118307861360, 5.33144196877749, 8.53631578218074,
    11.7060038949077, 14.8635881488839, 18.0155278304879,
    21.1643671187891, 24.3113254834588, 27.4570501848623,
    30.6019229722078, 33.7461812269726, 36.8899866873805,
    40.0334439409610, 43.1766274212415, 46.3195966792621,
    49.4623908440429, 52.6050411092602, 55.7475709551533,
    58.8900018651876, 62.0323477967829, 65.1746202084584,
    68.3168306640438, 71.4589869258787, 74.6010956133729,
    77.7431620631416, 80.8851921057280, 84.0271895462953,
    87.1691575709855, 90.3110993488875, 93.4530179063458,
    96.5949155953313, 99.7367932203820, 102.878653768715,
    106.020498619541, 109.162329055405, 112.304145672561,
    115.445950418834, 118.587744574512, 121.729527118091,
    124.871300497614, 128.013065217171, 131.154821965250,
    134.296570328107, 137.438311926144, 140.580047659913,
    143.721775748727, 146.863498476739, 150.005215971725,
    153.146928691331, 156.288635801966, 159.430338769213,
    162.572038308643, 165.713732347338, 168.855423073845,
    171.997111729391, 175.138794734935, 178.280475036977,
    181.422152668422, 184.563828222242, 187.705499575101])

def compute_GPDsum(am_r, pulse_duration, diffusion_time, diffusivity, radius):
    dam = diffusivity * am_r * am_r
    e11 = - dam * pulse_duration
    e2  = - dam * diffusion_time
    # dif = diffusion_time - pulse_duration
    e3 = e2-e11 #- np.outer(dam, dif)
    # plus = diffusion_time + pulse_duration
    e4 = e2+e11 #- np.outer(dam, plus)

    nom = -2 * e11 - 2 + \
          (2 * np.exp(e11)) + \
          (2 * np.exp(e2)) - np.exp(e3) - np.exp(e4)
    denom = dam ** 2 * am_r ** 2 * (radius ** 2 * am_r ** 2 - 1)
    return np.sum(nom / denom, axis=0)

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
        * (radius4**2 * am_r**2 - 1.0)
    )

    GPDsum = np.sum(nom / denom, axis=-1)  # sum over roots
    # shape: (n_radii, n_D, n_shells)

    return GPDsum