
import numpy as np

from cnn.params import CNNParam

class Adam:
	""" Adaptive Movement Estimation Algorithm 
	- combination of 'Gradient Descent with Momentum' and 'RMSprop' """
	
	ALIAS = 'adam'

	def __init__(self,learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8):
		self.ALPHA = learning_rate
		self.BETA1 = beta1	# Fist moment decay factor
		self.BETA2 = beta2	# Second moment decay factor
		self.EPSILON = epsilon	# This is a very small value just to avoid division by 0.

	def update_param(self,param) -> np.ndarray:
		# TODO: Change function sig. Needs to be consistent with other optimisers.
		if "momentum1" in param.associated_data:
			momentum1 = param.associated_data["momentum1"]
		else:
			momentum1 = np.zeros(shape=param.shape)
		if "momentum2" in param.associated_data:
			momentum2 = param.associated_data["momentum2"]
		else:
			momentum2 = np.zeros(shape=param.shape)

		momentum1 = self.BETA1 * momentum1 + (1 - self.BETA1) * param.gradient
		momentum2 = self.BETA2 * momentum2 + (1 - self.BETA2) * np.square(param.gradient)
		momentum1_hat = momentum1 / (1 - np.power(self.BETA1,self.model.iteration_index + 1))
		momentum2_hat = momentum2 / (1 - np.power(self.BETA2,self.model.iteration_index + 1))

		param.associated_data["momentum1"] = momentum1
		param.associated_data["momentum2"] = momentum2
		return param - self.ALPHA * ( momentum1_hat / (np.sqrt( momentum2_hat ) + self.EPSILON) )
