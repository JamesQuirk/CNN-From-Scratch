import numpy as np
from .layer import Layer

class Activation(Layer):
	def __init__(self,function: str=None,alpha=0.01,input_shape=None):
		super().__init__()

		self.LAYER_TYPE = self.__class__.__name__ + ' (' + function + ')'
		self.trainable = False
		self.alpha = alpha
		self.INPUT_SHAPE = input_shape

		self.FUNCTION = None if function is None else function.lower()

	def prepare_layer(self):
		if self.prev_layer is None:	# This means this is the first layer in the structure, so 'input' is the only thing before.
			assert self.INPUT_SHAPE is not None, 'ERROR: Must define input shape for first layer.'
		else:
			self.INPUT_SHAPE = self.prev_layer.OUTPUT_SHAPE
		self.OUTPUT_SHAPE = self.INPUT_SHAPE

	def _forwards(self,_input: np.ndarray) -> np.ndarray:
		if self.prev_layer.LAYER_TYPE == 'FC':
			assert len(_input.shape) == 2 and _input.shape[0] == self.INPUT_SHAPE[0], f'Expected input of shape {self.INPUT_SHAPE} instead got {(_input.shape[0],1)}'
		self.input = _input
		
		if self.FUNCTION is None:
			self.output = _input
		elif self.FUNCTION == 'relu':	# NOTE: This would work for Conv activation.
			# The ReLu function is highly computationally efficient but is not able to process inputs that approach zero or negative.
			self.output = np.maximum(_input,0)
		elif self.FUNCTION == 'softmax':
			assert self.prev_layer.LAYER_TYPE == 'FC', 'Softmax activation function is not supported for non-FC inputs.'
			# Softmax is a special activation function used for output neurons. It normalizes outputs for each class between 0 and 1, and returns the probability that the input belongs to a specific class.
			exp = np.exp(_input - np.max(_input,axis=0))	# Normalises by max value - provides "numerical stability"
			self.output = exp / np.sum(exp,axis=0)
			# print(_input)
			# print(self.output)
			# assert round(self.output.sum()) == 1, f'Output array sum {self.output.sum()} is not equal to 1.\nInput Array: {self.input.reshape((1,-1))}\nOuput Array: {self.output.reshape((1,-1))}'
		elif self.FUNCTION == 'sigmoid':	# NOTE: This would work for Conv activation.
			# The sigmoid function has a smooth gradient and outputs values between zero and one. For very high or low values of the input parameters, the network can be very slow to reach a prediction, called the vanishing gradient problem.
			self.output = 1 / (1 + np.exp(-_input))
		elif self.FUNCTION == 'step': # TODO: Define "step function" activation
			pass
		elif self.FUNCTION == 'tanh':
			# The TanH function is zero-centered making it easier to model inputs that are strongly negative strongly positive or neutral.
			self.output = ( np.exp(_input) - np.exp(-_input) ) / ( np.exp(_input) + np.exp(-_input) )
		elif self.FUNCTION == 'swish': # TODO: Define "Swish function" activation
			# Swish is a new activation function discovered by Google researchers. It performs better than ReLu with a similar level of computational efficiency.
			pass
		elif self.FUNCTION == 'leaky relu':
			# The Leaky ReLu function has a small positive slope in its negative area, enabling it to process zero or negative values.
			_input[_input <= 0] = self.alpha * _input[_input <= 0]
			self.output = _input
		elif self.FUNCTION == 'parametric relu': # TODO: Define "Parametric ReLu"
			#  The Parametric ReLu function allows the negative slope to be learned, performing backpropagation to learn the most effective slope for zero and negative input values.
			pass
		
		assert self.output.shape == _input.shape, f'Output shape, {self.output.shape}, not the same as input shape, {_input.shape}.'
		self._track_metrics(output=self.output)
		# print(f'Layer: {self.MODEL_STRUCTURE_INDEX} output:',self.output)
		return self.output

	def _backwards(self,dC_dA: np.ndarray) -> np.ndarray:
		"""Compute derivative of Activation w.r.t. Z
		NOTE: CURRENTLY NOT SUPPORTED FOR CONV/POOL LAYERS.
		"""
		assert dC_dA.shape == self.output.shape, f'dC/dA shape, {dC_dA.shape}, not as expected, {self.output.shape}.'
		self._track_metrics(cost_gradient=dC_dA)
		dA_dZ = np.zeros(shape=(self.output.shape[1],self.output.shape[0],self.prev_layer.output.shape[0]))	# TODO: Will need varifying for Conv Activation.
		if self.FUNCTION is None: # a = z
			dA_dZ = np.broadcast_to(np.diag(np.ones(dA_dZ.shape[-1])),dA_dZ.shape )
		elif self.FUNCTION == 'relu':
			# Insert layer input along dA_dZ diagonals - values > 0 -> 1; values <= 0 -> 0
			ix,iy = np.diag_indices_from(dA_dZ[0,:,:])
			dA_dZ[:,iy,ix] = (self.input.T > 0).astype(int)
		elif self.FUNCTION == 'softmax':
			# Vectorised implementation from https://stackoverflow.com/questions/59286911/vectorized-softmax-gradient
			# NOTE: Transpose is required to create the square matrices of each set of node values.
			outputT = self.output.T
			diag_matrices = outputT.reshape(outputT.shape[0],-1,1) * np.diag(np.ones(outputT.shape[1]))	# Diagonal Matrices
			outer_product = np.matmul(outputT.reshape(outputT.shape[0],-1,1), outputT.reshape(outputT.shape[0],1,-1))	# Outer product
			Jsm = diag_matrices - outer_product
			dA_dZ = Jsm	# NOTE: Even though this equation uses softmax transpose at start, the output does not require transposing because the softmax derivative is symmetrical along diagonal.

		elif self.FUNCTION == 'sigmoid':
			# sig (1 - sig) across diagonals
			ix,iy = np.diag_indices_from(dA_dZ[0,:,:])
			dA_dZ[:,iy,ix] = (self.output * (1 - self.output)).T	# Element-wise multiplication.
		elif self.FUNCTION == 'step': # TODO: Define "step function" derivative
			dA_dZ = None
		elif self.FUNCTION == 'tanh':
			dA_dZ = np.diag((1 - np.square( self.output )).flatten())
		elif self.FUNCTION == 'swish': # TODO: Define "Swish function" derivative
			dA_dZ = None
		elif self.FUNCTION == 'leaky relu':
			ix,iy = np.diag_indices_from(dA_dZ[0,:,:])
			dA_dZ[:,iy,ix] = ( (self.input > 0).astype(int) + ((self.input < 0).astype(int) * self.alpha ) ).T

			# input_diag = np.diag(self.input.flatten())
			# input_diag[input_diag > 0] = 1
			# input_diag[input_diag < 0] = self.alpha
			# dA_dZ = input_diag
		elif self.FUNCTION == 'parametric relu': # TODO: Define "Parametric ReLu" derivative
			dA_dZ = None
		
		assert dA_dZ is not None, f'No derivative defined for chosen activation function "{self.FUNCTION}"'
		assert dA_dZ.shape[1:] == (self.output.shape[0],self.output.shape[0]), 'dA/dZ is expected to be a square matrix (for each example in batch) containing gradient between each activation node and each input node.'
		# print('Layer: ', self.LAYER_TYPE)
		# print('Local gradient shape:',dA_dZ.shape)
		# print('Cost gradient shape:',dC_dA.shape)

		dC_dAexpanded = dC_dA.T.reshape((dC_dA.T.shape[0],-1,1))
		dC_dZexpanded = np.matmul(dA_dZ,dC_dAexpanded)
		dC_dZ = dC_dZexpanded.reshape(dC_dA.shape[1],-1).T
		
		assert dC_dZ.shape == self.prev_layer.output.shape, f'Back propagating dC_dZ has shape: {dC_dZ.shape} when previous layer output has shape {self.prev_layer.output.shape}'
		if self.FUNCTION is None:
			assert np.array_equal(dC_dZ,dC_dA), 'For activation: None; dC/dZ is expected to be the same as dC/dA.'
		
		return dC_dZ
