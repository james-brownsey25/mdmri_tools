import numpy as np
from scipy import special
from scipy.special import i0  # I_0 Bessel
from scipy.integrate import quad
from .base import Parameter, BaseCompartment
from .utils import normalize_shells, normalize_TE, normalize_gradients, calc_bval
from typing import Any

class Tensor(BaseCompartment):
    """
    Represents water with gaussian diffusion in 3D.
    The tensor is assumed to be aligned along the z-axis ([0,0,1]).

    """
    name = 'tensor'
    parameters = (
        Parameter("D_ax", unit='um^2/ms', required=True),
        Parameter("D_perp1", unit='um^2/ms', required=True),
        Parameter("D_perp2", unit='um^2/ms', required=True),
        Parameter("T2", unit="ms", required=False, default_value=None),
    )

    def predict(self, **params: Any) -> np.ndarray:
        """
        Expected keys in params:
            - shells (required): scalar, 1D, or (3, N) array
            - gradients (required): (3, N) or (N, 3) array
            - TE (optional): scalar or 1D

        Also includes microstructural params:
            - D_ax
            - D_perp1
            - D_perp2
            - T2 (may be None)
        """
        # ---- 1. Required 'shells' argument ----
        if "shells" not in params:
            raise ValueError(
                f"'shells' argument is required in predict() for {self.name}"
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
        D_ax = self.values.get("D_ax",None)
        D_perp1 = self.values.get("D_perp1",None)
        D_perp2 = self.values.get("D_perp2",None)

        if D_ax is None or D_perp1 is None or D_perp2 is None:
            raise ValueError(
                f"Missing microstructural parameters D_ax, D_perp1 and/or D_perp2 for compartment '{self.name}'. "
                f"Pass them when instantiating the object."
                )

        # ---- 5. Compute the signal shape ----
        # Internal axis order: (n_shells, n_TE, n_dirs)
        # If TE_arr is None: treat n_TE = 1 but you can omit decay or set TE=0

        signal = np.zeros((n_shells, n_TE, n_dirs), dtype=float)

        # --- Case 1: shells_arr is 1D → b-values
        if shells_arr.ndim == 1:
            for i_shell in range(n_shells):
                b_val = shells_arr[i_shell]
                for i_TE in range(n_TE):
                    TE_val = TE_arr[i_TE] if TE_arr is not None else None
                    for i_dir in range(n_dirs):
                        bvec = grad_arr[i_dir]
                        bv2 = bvec * bvec
                        attenuation = - b_val * np.sum([bv2[..., idx] * param for idx, param in enumerate([D_perp2,D_perp1,D_ax])], 0)
                        if T2 is None:
                            signal[i_shell,i_TE,i_dir] = np.exp(attenuation)
                        else:
                            T2_decay = np.exp(-TE_val/T2)
                            signal[i_shell,i_TE,i_dir] = np.exp(attenuation) * T2_decay

        # --- Case 2: shells_arr is (3, N) → [G, Δ, δ]
        elif shells_arr.ndim == 2:
            for i_shell in range(n_shells):
                G = shells_arr[0, i_shell]
                Delta = shells_arr[1, i_shell]
                delta = shells_arr[2, i_shell]
                b_val = calc_bval(G, delta, Delta)  # note arg order in your calc_bval

                for i_TE in range(n_TE):
                    TE_val = TE_arr[i_TE] if TE_arr is not None else None
                    for i_dir in range(n_dirs):
                        bvec = grad_arr[i_dir]
                        bv2 = bvec * bvec
                        attenuation = - b_val * np.sum([bv2[..., idx] * param for idx, param in enumerate([D_perp2,D_perp1,D_ax])], 0)
                        if T2 is None:
                            signal[i_shell,i_TE,i_dir] = np.exp(attenuation)
                        else:
                            T2_decay = np.exp(-TE_val/T2)
                            signal[i_shell,i_TE,i_dir] = np.exp(attenuation) * T2_decay

        else:
            raise RuntimeError(
                f"normalize_shells returned unexpected ndim={shells_arr.ndim}"
            )

        return np.squeeze(signal)
    
    def predict_mean(self, **params: Any) -> np.ndarray:
        """
        Expected keys in params:
            - shells (required): scalar, 1D, or (3, N) array
            - TE (optional): scalar or 1D

        Also includes microstructural params:
            - D_ax
            - D_perp1
            - D_perp2
            - T2 (may be None)
        """

        # ---- 1. Required 'shells' argument ----
        if "shells" not in params:
            raise ValueError(
                f"'shells' argument is required in predict_mean() for {self.name}"
            )
        shells_raw = params["shells"]
        shells_arr, n_shells = normalize_shells(shells_raw)

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
        D_ax = self.values.get("D_ax",None)
        D_perp1 = self.values.get("D_perp1",None)
        D_perp2 = self.values.get("D_perp2",None)

        if D_ax is None or D_perp1 is None or D_perp2 is None:
            raise ValueError(
                f"Missing microstructural parameters D_ax, D_perp1 and/or D_perp2 for compartment '{self.name}'. "
                f"Pass them when instantiating the object."
                )

        # ---- 5. Compute the signal shape ----
        # Internal axis order: (n_shells, n_TE, n_dirs)
        # If TE_arr is None: treat n_TE = 1 but you can omit decay or set TE=0

        signal = np.zeros((n_shells, n_TE), dtype=float)

        # --- Case 1: shells_arr is 1D → b-values
        if shells_arr.ndim == 1:
            for i_shell in range(n_shells):
                b_val = shells_arr[i_shell]
                for i_TE in range(n_TE):
                    TE_val = TE_arr[i_TE] if TE_arr is not None else None
                    attenuation = powder_average_general_tensor(b_val, D_perp1, D_perp2, D_ax)
                    if T2 is None:
                        signal[i_shell,i_TE] = attenuation
                    else:
                        T2_decay = np.exp(-TE_val/T2)
                        signal[i_shell,i_TE] = attenuation * T2_decay

        # --- Case 2: shells_arr is (3, N) → [G, Δ, δ]
        elif shells_arr.ndim == 2:
            for i_shell in range(n_shells):
                G = shells_arr[0, i_shell]
                Delta = shells_arr[1, i_shell]
                delta = shells_arr[2, i_shell]
                b_val = calc_bval(G, delta, Delta)  # note arg order in your calc_bval

                for i_TE in range(n_TE):
                    TE_val = TE_arr[i_TE] if TE_arr is not None else None
                    attenuation = powder_average_general_tensor(b_val, D_perp1, D_perp2, D_ax)
                    if T2 is None:
                        signal[i_shell,i_TE] = attenuation
                    else:
                        T2_decay = np.exp(-TE_val/T2)
                        signal[i_shell,i_TE] = attenuation * T2_decay

        else:
            raise RuntimeError(
                f"normalize_shells returned unexpected ndim={shells_arr.ndim}"
            )

        return np.squeeze(signal)

def powder_average_general_tensor(b, D1, D2, D3):
    lam = 0.5 * (D1 + D2)
    Delta = 0.5 * (D1 - D2)
    prefactor = np.exp(-b * D3)
    
    def integrand(u):
        # u in [0,1]
        one_minus_u2 = 1.0 - u**2
        return np.exp(-b * (lam - D3) * one_minus_u2) * i0(b * Delta * one_minus_u2)
    
    integral, _ = quad(integrand, 0.0, 1.0, epsabs=1e-10, epsrel=1e-8)
    return prefactor * integral