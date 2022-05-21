from cnn.params import CNNParam
import numpy as np


class BaseOptimiser:
	ALIAS = 'base'
	def update_param(param: CNNParam) -> np.ndarray:
		raise NotImplementedError("Optimisers inheriting from this base class must implement update_param() method.")
