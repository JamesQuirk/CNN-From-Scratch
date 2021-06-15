
import numpy as np

class Adam:
	""" Adaptive Movement Estimation Algorithm 
	- combination of 'Gradient Descent with Momentum' and 'RMSprop' """
	
	ALIAS = 'adam'

	def __init__(self,learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8):
		self.ALPHA = learning_rate
		self.BETA1 = beta1	# Fist moment decay factor
		self.BETA2 = beta2	# Second moment decay factor
		self.EPSILON = epsilon	# This is a very small value just to avoid division by 0.

	@property
	def model(self):
		return self._model

	@model.setter
	def model(self,val):
		self._model = val
		self._init_params()

	def _init_params(self):
		self.params = {}	# Adam Params; contains m, v for each trainable parameter.
		for layer in self.model.structure:
			if layer.TRAINABLE:
				self.params[layer.MODEL_STRUCTURE_INDEX] = {}
				for param_name, param in layer.params.items():
					self.params[layer.MODEL_STRUCTURE_INDEX][param_name] = {
						'm':np.zeros(shape=param['values'].shape),
						'v':np.zeros(shape=param['values'].shape)
					}

	def update_param(self,param,param_grad,layer_index) -> np.ndarray:
		# TODO: Change function sig. Needs to be consistent with other optimisers.
		assert param['name'] in ('weights','filters','bias'), 'Invalid param name provided.'
		moment1 = self.params[layer_index][param['name']]['m']
		moment2 = self.params[layer_index][param['name']]['v']

		moment1 = self.BETA1 * moment1 + (1 - self.BETA1) * param_grad
		moment2 = self.BETA2 * moment2 + (1 - self.BETA2) * np.square(param_grad)
		moment1_hat = moment1 / (1 - np.power(self.BETA1,self.model.iteration_index + 1))
		moment2_hat = moment2 / (1 - np.power(self.BETA2,self.model.iteration_index + 1))

		self.params[layer_index][param['name']]['m'] = moment1
		self.params[layer_index][param['name']]['v'] = moment2
		return param['values'] - self.ALPHA * ( moment1_hat / (np.sqrt( moment2_hat ) + self.EPSILON) )
