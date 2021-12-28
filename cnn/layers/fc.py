import numpy as np

from cnn.params import CNNParam
from .layer import Layer
from cnn import utils

class FC(Layer):
	"""
	The Fully Connected Layer is defined as being the layer of nodes and the weights of the connections that link 
	those nodes to the previous layer.
	"""
	def __init__(self, num_nodes, activation: str=None,random_seed=42,initiation_method=None,input_shape=None,track_history=True):
		"""
		- n: Number of nodes in layer.
		- activation: The name of the activation function to be used. The activation is handled by an Activation object that is transparent to the user here. Defaults to None - a transparent Activation layer will still be added however, the data passing through will be untouched.
		- initiation_method (str): This is the method used to initiate the weights and biases. Options: "kaiming", "xavier" or None. Default is none - this simply takes random numbers from standard normal distribution with no scaling.
		- input_shape (tuple): Input shape for a single example. (n,1) where n is number of input nodes (features).
		"""
		super().__init__()

		self.LAYER_TYPE = self.__class__.__name__
		self.TRAINABLE = True
		self.NUM_NODES = num_nodes
		self.ACTIVATION = None if activation is None else activation.lower()
		self.RANDOM_SEED = random_seed
		self.INITIATION_METHOD = None if initiation_method is None else initiation_method.lower()
		if input_shape is not None:
			assert len(input_shape) == 2 and input_shape[1] == 1, 'Invalid input_shape tuple. Expected (n,1)'
		self.INPUT_SHAPE = input_shape
		self.TRACK_HISTORY = track_history

	def prepare_layer(self) -> None:
		if self.prev_layer is None:	# This means this is the first layer in the structure, so 'input' is the only thing before.
			assert self.INPUT_SHAPE is not None, 'ERROR: Must define input shape for first layer.'
		else:
			self.INPUT_SHAPE = self.prev_layer.OUTPUT_SHAPE
		
		self.weights = utils.array.array_init(shape=(self.NUM_NODES,self.INPUT_SHAPE[0]),method=self.INITIATION_METHOD,seed=self.RANDOM_SEED)	# NOTE: this is the correct orientation for vertical node array.

		self.bias = np.zeros(shape=(self.NUM_NODES,1))	# NOTE: Recommended to initaite biases to zero.

		self.OUTPUT_SHAPE = (self.NUM_NODES,1)


	def _forwards(self,_input: np.ndarray) -> np.ndarray:
		if self.prev_layer is None:
			self.input = _input
		else:
			assert len(_input.shape) == 2 and _input.shape[0] == self.INPUT_SHAPE[0], f'Expected input of shape {self.INPUT_SHAPE} instead got {(_input.shape[0],1)}'
			self.input = _input

		self.output = np.dot( self.weights, self.input ) + self.bias
		
		assert len(self.output.shape) == 2 and self.output.shape[0] == self.OUTPUT_SHAPE[0], f'Output shape, {(self.output.shape[0],1)}, not as expected, {self.OUTPUT_SHAPE}'
		if self.TRACK_HISTORY: self._track_metrics(output=self.output)
		return self.output

	def _backwards(self, dC_dZ: np.ndarray) -> np.ndarray:
		"""
		Take cost gradient dC/dZ (how the output of this layer affects the cost) and backpropogate

		Z = W . I + B

		"""
		assert dC_dZ.shape == self.output.shape, f'dC/dZ shape, {dC_dZ.shape}, does not match Z shape, {self.output.shape}.'
		if self.TRACK_HISTORY: self._track_metrics(cost_gradient=dC_dZ)

		dZ_dW = self.input.T	# Partial diff of weighted sum (Z) w.r.t. weights
		dZ_dB = 1
		dZ_dI = self.weights.T	# Partial diff of weighted sum w.r.t. input to layer.
		
		dC_dW = np.dot(dC_dZ,dZ_dW)
		assert dC_dW.shape == self.weights.shape, f'dC/dW shape {dC_dW.shape} does not match W shape {self.weights.shape}'
		self.weights.gradient = dC_dW
		if self.weights.trainable:
			self.weights = self.model.OPTIMISER.update_param(self.weights)

		dC_dB = np.sum(dC_dZ * dZ_dB, axis=1,keepdims=True)	# Element-wise multiplication (dZ_dB turns out to be just 1)

		assert dC_dB.shape == self.bias.shape, f'dC/dB shape {dC_dB.shape} does not match B shape {self.bias.shape}'
		self.bias.gradient = dC_dB
		if self.bias.trainable:
			self.bias = self.model.OPTIMISER.update_param(self.bias)

		dC_dI = np.dot( dZ_dI , dC_dZ )
		assert dC_dI.shape == self.input.shape, f'dC/dI shape {dC_dI.shape} does not match input shape {self.input.shape}.'
		return dC_dI

	@property
	def weights(self):
		return self._weights

	@weights.setter
	def weights(self,value):
		self._weights = CNNParam(value)

	@property
	def bias(self):
		return self._bias

	@bias.setter
	def bias(self,value):
		self._bias = CNNParam(value)
