from ..layer import Layer

class BaseActivation(Layer):
	def __init__(self,input_shape=None):
		super().__init__()

		self.trainable = False
		self.INPUT_SHAPE = input_shape

	def prepare_layer(self):
		if self.prev_layer is None:	# This means this is the first layer in the structure, so 'input' is the only thing before.
			assert self.INPUT_SHAPE is not None, 'ERROR: Must define input shape for first layer.'
		else:
			self.INPUT_SHAPE = self.prev_layer.OUTPUT_SHAPE
		self.OUTPUT_SHAPE = self.INPUT_SHAPE

	def forwards(self, X):
		if self.prev_layer.LAYER_TYPE == 'FC':
			assert len(X.shape) == 2 and X.shape[0] == self.INPUT_SHAPE[0], f'Expected input of shape {self.INPUT_SHAPE} instead got {(X.shape[0],1)}'
		self.input = X

		self._forwards(X)

		assert self.output.shape == X.shape, f'Output shape, {self.output.shape}, not the same as input shape, {X.shape}.'
		self._track_metrics(output=self.output)

		return self.output

	def backwards(self, dCdA):
		assert dCdA.shape == self.output.shape, f'dC/dA shape, {dCdA.shape}, not as expected, {self.output.shape}.'
		self._track_metrics(cost_gradient=dCdA)

		dCdZ = self._backwards(dCdA)

		assert dCdZ.shape == self.prev_layer.output.shape, f'Back propagating dC_dZ has shape: {dCdZ.shape} when previous layer output has shape {self.prev_layer.output.shape}'

		return dCdZ



