import numpy as np
from .layer import Layer


class Flatten(Layer):
	""" A psuedo layer that simply adjusts the data dimension as it passes between 2D Conv or Pool layers to the 1D FC layers. 
	The output shapes of the Conv/ Pool layer is expected to be (m,c,h,w) which is converted to (n,m)."""

	def __init__(self,input_shape=None):
		"""
		input_shape (tuple): Shape of a single example eg (channels,height,width). Irrespective of batch size. 
		"""

		super().__init__()

		self.trainable = False
		if input_shape is not None:
			assert len(input_shape) == 3, f'ERROR: Expected input_shape to be a tuple of length 3; (channels, height, width).'
		self.INPUT_SHAPE = input_shape

	def prepare_layer(self) -> None:
		if self.prev_layer is None:	# This means this is the first layer in the structure, so 'input' is the only thing before.
			assert self.INPUT_SHAPE is not None, 'ERROR: Must define input shape for first layer.'
		else:
			self.INPUT_SHAPE = self.prev_layer.OUTPUT_SHAPE
		self.OUTPUT_SHAPE = (np.prod(self.INPUT_SHAPE),1)	# Output shape for a single example.
		# self.output = np.zeros(shape=(np.prod(self.INPUT_SHAPE[1:]),self.INPUT_SHAPE[0]))

	def _forwards(self,_input: np.ndarray) -> np.ndarray:
		assert _input.shape[1:] == self.INPUT_SHAPE, f'ERROR:: Input has unexpected shape: {_input.shape[1:]} | expected: {self.INPUT_SHAPE}'
		self.input = _input
		self.output = _input.T.reshape((-1,_input.shape[0]))	# Taking transpose here puts each example into its own column - number of columns == number of examles.

		self._track_metrics(output=self.output)
		return self.output	# NOTE: 2D matrix, (number of nodes, number of examples in batch)

	def _backwards(self,cost_gradient):
		"""
		cost_gradient expected to have shape (n,m) where n == channels * height * width
		"""
		self._track_metrics(cost_gradient=cost_gradient)
		return cost_gradient.reshape(self.input.T.shape).T	# Outputs shape (m,c,h,w)
