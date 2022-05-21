from ..layer import Layer

class BaseActivation(Layer):
	def __init__(self,function: str=None,alpha=0.01,input_shape=None):
		super().__init__()

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

