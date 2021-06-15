import numpy as np

class RMSProp:
	''' Root mean square propagation '''

	ALIAS = 'rmsprop'

	def __init__(self,learning_rate=0.001,beta=0.9,epsilon=1e-8):
		self.ALPHA = learning_rate
		self.EPSILON = epsilon
		self.BETA = beta

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
						's':np.zeros(shape=param['values'].shape)
					}

	def update_param(self,param,param_grad,layer_index) -> np.ndarray:
		# TODO: Change function sig. Needs to be consistent with other optimisers.
		assert param['name'] in ('weights','filters','bias'), 'Invalid param name provided.'
		s = self.params[layer_index][param['name']]['s']

		s = self.BETA * s + (1 - self.BETA) * np.square(param_grad)

		self.params[layer_index][param['name']]['s'] = s
		return param['values'] - self.ALPHA * ( param_grad / (np.sqrt( s ) + self.EPSILON) )
