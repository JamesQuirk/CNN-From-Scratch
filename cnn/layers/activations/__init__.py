from .relu import ReLU, LeakyReLU
from .softmax import Softmax
from .sigmoid import Sigmoid
from .tanh import Tanh

# ------------- BELOW IS DYNAMIC TO AVAILABLE ACTIVATION CLASSES ----------------

# Expose list of all activation class names.
import inspect
import sys
available_activations = [c[0] for c in inspect.getmembers(sys.modules[__name__], inspect.isclass)]

__activation_classes = [c[1] for c in inspect.getmembers(sys.modules[__name__], inspect.isclass)]

def from_name(name):
	for activation in __activation_classes:
		if activation.ALIAS == name or activation.__name__ == name:
			return activation()
