
from cnn.optimisers import BaseOptimiser
from cnn.params import CNNParam
import numpy as np

class GradientDescent(BaseOptimiser):

	ALIAS = 'gd'

	def __init__(self,learning_rate=0.001,beta=0.9,):
		self.ALPHA = learning_rate

	def update_param(self,param: CNNParam) -> np.ndarray:
		return param - self.ALPHA * param.gradient
