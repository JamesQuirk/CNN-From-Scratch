from .activation import Activation
from .conv import Conv2D
from .fc import FC
from .flatten import Flatten
from .pool import Pool


# Expose list of all optimiser class names.
import inspect
import sys
layers = [c[0] for c in inspect.getmembers(sys.modules[__name__], inspect.isclass)]