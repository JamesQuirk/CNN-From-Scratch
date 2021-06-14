from .adam import Adam
from .gd import GradientDescent
# from .rmsprop import RMSProp



# Expose list of all optimiser class names.
import inspect
import sys
optimisers = [c[0] for c in inspect.getmembers(sys.modules[__name__], inspect.isclass)]
