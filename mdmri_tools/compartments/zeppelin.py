import numpy as np
from scipy import special
from .base import Parameter, BaseCompartment
from .compartment_utils import normalize_shells, normalize_TE, normalize_gradients, calc_bval
from typing import Any

class Zeppelin(BaseCompartment):
    """
    Represents water with gaussian diffusion in 3D.
    The zeppelin is assumed to be aligned along the z-axis ([0,0,1]).

    """
    name = 'zeppelin'
    parameters = (
        Parameter("D_ax", unit='um^2/ms', required=True),
        Parameter("D_perp", unit='um^2/ms', required=True),
        Parameter("T2", unit="ms", required=False, default_value=None),)

    def predict(self, **params: Any) -> np.ndarray:
        """
        Expected keys in params:
            - shells (required): scalar, 1D, or (3, N) array
            - gradients (required): (3, N) or (N, 3) array
            - TE (optional): scalar or 1D

        Also includes microstructural params:
            - D_ax
            - D_perp
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
        D_perp = self.values.get("D_perp",None)

        if D_ax is None or D_perp is None:
            raise ValueError(
                f"Missing microstructural parameters D_ax and/or D_perp for compartment '{self.name}'. "
                f"Pass them when instantiating the object."
                )

        # ---- 5. Compute the signal shape ----
        # Internal axis order: (n_shells, n_TE, n_dirs)
        # If TE_arr is None: treat n_TE = 1 but you can omit decay or set TE=0

        D_ax = np.atleast_1d(D_ax).astype(float)
        D_perp = np.atleast_1d(D_perp).astype(float)

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

        n_Daxs = D_ax.size
        n_Dperps = D_perp.size

        D_ax_full = D_ax[:, None, None, None, None, None]
        D_perp_full = D_perp[None, :, None, None, None, None]

        if shells_arr.ndim == 1:
            b_b = shells_arr[None, None, None, :, None, None]      # same

        elif shells_arr.ndim == 2:
            G_all, Delta_all, delta_all = shells_arr
            b_all = calc_bval(G_all, delta_all, Delta_all) 
            b_b = b_all[None, None, None, :, None, None]

        bz2 = grad_arr[:,2]**2
        bz2_full = bz2[None, None, None, None, None, :]

        signal = np.exp(-D_ax_full * b_b * bz2_full) * np.exp(-D_perp_full * b_b * (1 - bz2_full))                    

        if T2_arr is not None:
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
            # (1,1,n_T2,1,n_TE,1)

            signal = signal * T2_decay_b

        return np.squeeze(signal)
    
    def predict_mean(self, **params: Any) -> np.ndarray:
        """
        Expected keys in params:
            - shells (required): scalar, 1D, or (3, N) array
            - TE (optional): scalar or 1D

        Also includes microstructural params:
            - D_ax
            - D_perp
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
        D_perp = self.values.get("D_perp",None)

        if D_ax is None or D_perp is None:
            raise ValueError(
                f"Missing microstructural parameters D_ax and/or D_perp for compartment '{self.name}'. "
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
                    D_iso = (D_ax + 2 * D_perp)/3
                    D_delta = (D_ax - D_perp)/(D_ax + 2*D_perp)
                    alpha = 3 * b_val * D_iso * D_delta
                    attenuation = np.exp(-b_val * D_iso * (1-D_delta)) * H(alpha)
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
                    D_iso = (D_ax + 2 * D_perp)/3
                    D_delta = (D_ax - D_perp)/(D_ax + 2*D_perp)
                    alpha = 3 * b_val * D_iso * D_delta
                    attenuation = np.exp(-b_val * D_iso * (1-D_delta)) * H(alpha)
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
        
def H(alpha):
    a = np.asarray(alpha)
    out = np.empty_like(a, dtype=float)

    pos = a > 0
    x = np.sqrt(a[pos])
    out[pos] = np.sqrt(np.pi) / (2*x) * special.erf(x)

    z = a == 0
    out[z] = 1.0

    neg = a < 0
    if np.any(neg):
        a_neg = -a[neg]
        y = np.sqrt(a_neg)
        out[neg] = np.sqrt(np.pi) / (2*y) * special.erfi(y)

    return out