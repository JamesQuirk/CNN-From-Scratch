
from cnn.params import CNNParam


class GradientDescent:

	ALIAS = 'gd'

	def __init__(self,learning_rate=0.001,beta=0.9,):
		self.ALPHA = learning_rate

	def update_param(self,param):
		return param - self.ALPHA * param.gradient
