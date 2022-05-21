from cnn.params import CNNParam
from .base import BaseOptimiser
from .adam import Adam
from .gd import GradientDescent
from .rmsprop import RMSProp
import numpy as np

# ------------- BELOW IS DYNAMIC TO AVAILABLE OPTIMISER CLASSES ----------------

# Expose list of all optimiser class names.
import inspect
import sys
__optimiser_classes = [c[1] for c in inspect.getmembers(sys.modules[__name__], lambda cls: inspect.isclass(cls) and issubclass(cls,BaseOptimiser))]

# Following includes both class name and alias property.
optimiser_identifiers = [c[0] for c in inspect.getmembers(sys.modules[__name__], inspect.isclass)] + [opt.ALIAS for opt in __optimiser_classes]

def from_name(name):
	for optimiser in __optimiser_classes:
		if optimiser.ALIAS == name or optimiser.__name__ == name:
			return optimiser()
			

