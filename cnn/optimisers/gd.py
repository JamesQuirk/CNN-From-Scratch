
class GradientDescent:
	def __init__(self,learning_rate=0.001,beta=0.9,):
		self.ALPHA = learning_rate

	def update_param(self,param,param_grad,*args):
		return param['values'] - self.ALPHA * param_grad
