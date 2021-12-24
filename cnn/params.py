import numpy as np


class CNNParam(np.ndarray):
	def __new__(self, *args, trainable=True, **kwargs) -> None:
		arr = np.array(*args,**kwargs)
		obj = np.asarray(arr).view(self)
		
		obj.trainable = trainable
		obj._cost_gradient = None
		obj.associated_data = {}	# This can be used to containing things like momentum values for optimisers.

		return obj

	@property
	def gradient(self):
		return self._cost_gradient

	@gradient.setter
	def gradient(self, value):
		self._cost_gradient = value
