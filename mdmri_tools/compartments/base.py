"""
Contains classes describing individual water compartments within the tissue
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Collection, Union
from dataclasses import dataclass
from typing import Union, Optional, Any, Dict

Number = Union[int, float]

@dataclass(frozen=True)
class Parameter:
    """
    Represents a single microstructural tissue parameter.

    Returns:
        [type]: [description]
    """
    name: str
    unit: str = ""
    required: bool = True
    default_value: Optional[Number] = None 

    def __post_init__(self):
        # if parameter is required, it should not have a default
        if self.required and self.default_value is not None:
            raise ValueError(
                f"Parameter '{self.name}' is marked required but has a default_value."
            )
        
    @property
    def title(self) -> str:
        return self.name + (f" ({self.unit})" if self.unit else "")

class BaseCompartment(ABC):
    """
    Abstract base class for all compartments.
    """
    name: str = ""
    # Subclasses must define this as a tuple/list of Parameter
    parameters: Collection[Parameter] = ()

    def __init__(self, **kwargs: Any):
        """
        Initialize the compartment with its microstructural parameters.

        Example:
            Ball(diffusivity=3.0, T2=80.0)

        Required parameters must be supplied.
        Optional parameters may be omitted (and fall back to their defaults).
        """
        # Validate and set microstructural parameter values
        self.values: Dict[str, Any] = {}

        provided_keys = set(kwargs.keys())
        valid_param_names = set(self.parameter_names)

        # Check for unknown parameters
        unknown = provided_keys - valid_param_names
        if unknown:
            raise ValueError(
                f"Unrecognized parameters for {self.__class__.__name__}: {unknown}"
            )
        
        # Assign values, checking required/optional
        for param in self.parameters:
            if param.required:
                if param.name not in kwargs:
                    raise ValueError(
                        f"Missing required parameter '{param.name}' "
                        f"for {self.__class__.__name__}"
                    )
                self.values[param.name] = kwargs[param.name]
            else:
                # Optional parameter: use provided value or default
                if param.name in kwargs:
                    self.values[param.name] = kwargs[param.name]
                else:
                    self.values[param.name] = param.default_value
        
    # ---------------- Introspection helpers ------------------- #

    @property
    def parameter_names(self, ):
        """
        All the parameter names.
        """
        return [p.name for p in self.parameters]

    @property
    def parameter_titles(self,):
        """ 
        All parameter names with their units.
        """
        return [p.title for p in self.parameters]

    @abstractmethod
    def predict(self, **parameters: Any) -> np.ndarray:
        """
        Predicts signal based on user-provided parameters and acquisition parameters
        """
        raise NotImplementedError()

    def combine_params(self, **acquisition_params: Any) -> Dict[str, Any]:
        """
        Merge microstructural parameters with acquisition parameters.
        """
        combined = dict(self.values)
        combined.update(acquisition_params)
        return combined

    def __call__(self, **acquisition_params: Any) -> np.ndarray:
        combined = self.combine_params(**acquisition_params)
        return self.predict(**combined)

    def __repr__(self, ) -> str:
        args = ', '.join(f"{k}={v}" for k, v in self.values.items())
        return f"{self.__class__.__name__}({args})"