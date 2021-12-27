import numpy as np
from cnn.optimisers import BaseOptimiser

from cnn.params import CNNParam

class RMSProp(BaseOptimiser):
	''' Root mean square propagation '''

	ALIAS = 'rmsprop'

	def __init__(self,learning_rate=0.001,beta=0.9,epsilon=1e-8):
		self.ALPHA = learning_rate
		self.EPSILON = epsilon
		self.BETA = beta

	def update_param(self,param: CNNParam) -> np.ndarray:
		if "momentum1" in param.associated_data["momentum1"]:
			s = param.associated_data["momentum1"]
		else:
			s = np.zeros(shape=param.shape)

		s = self.BETA * s + (1 - self.BETA) * np.square(param.gradient)

		param.associated_data["momentum1"] = s
		return param - self.ALPHA * ( param.gradient / (np.sqrt( s ) + self.EPSILON) )
