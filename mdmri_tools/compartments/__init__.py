from typing import List, Type

from .ball import Ball
from .stick import Stick
from .sphere import Sphere
from .base import BaseCompartment
from .tensor import Tensor
from .zeppelin import Zeppelin
from .cylinder import Cylinder

def list_compartments() -> List[Type[BaseCompartment]]:
    """
    Returns list of available compartments
    """
    return [Stick, Ball, Tensor, Sphere, Zeppelin, Cylinder]

__all__ = ["Ball","Stick","Sphere","Tensor","Zeppelin","Cylinder","BaseCompartment","list_compartments"]