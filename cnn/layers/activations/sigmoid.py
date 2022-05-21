import numpy as np
from .base import BaseActivation

class Sigmoid(BaseActivation):
	
	def _forwards(self,X:np.ndarray):
		self.input = X.copy()
		# The sigmoid function has a smooth gradient and outputs values between zero and one. For very high or low values of the input parameters, the network can be very slow to reach a prediction, called the vanishing gradient problem.
		self.output = 1 / (1 + np.exp(-X))
		return self.output

	def _backwards(self,dCdA:np.ndarray):
		# Init dAdZ as square array representing all connections between input and output nodes
		dAdZ = np.zeros(shape=(self.output.shape[1],self.output.shape[0],self.prev_layer.output.shape[0]))	# TODO: Will need varifying for Conv Activation.

		# sig (1 - sig) across diagonals
		ix,iy = np.diag_indices_from(dAdZ[0,:,:])
		dAdZ[:,iy,ix] = (self.output * (1 - self.output)).T	# Element-wise multiplication.

		dC_dAexpanded = dCdA.T.reshape((dCdA.T.shape[0],-1,1))
		dC_dZexpanded = np.matmul(dAdZ,dC_dAexpanded)

		return dC_dZexpanded.reshape(dCdA.shape[1],-1).T
