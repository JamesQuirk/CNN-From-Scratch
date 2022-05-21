import numpy as np
from .base import BaseActivation

class ReLU(BaseActivation):
	
	def _forwards(self,X:np.ndarray):
		self.input = X.copy()
		self.output = np.maximum(self.input,0)
		return self.output

	def _backwards(self,dCdA:np.ndarray):
		# Init dAdZ as square array representing all connections between input and output nodes
		dAdZ = np.zeros(shape=(self.output.shape[1],self.output.shape[0],self.prev_layer.output.shape[0]))	# TODO: Will need varifying for Conv Activation.

		# Insert layer input along dAdZ diagonals - values > 0 -> 1; values <= 0 -> 0
		ix,iy = np.diag_indices_from(dAdZ[0,:,:])
		dAdZ[:,iy,ix] = (self.input.T > 0).astype(int)

		dC_dAexpanded = dCdA.T.reshape((dCdA.T.shape[0],-1,1))
		dC_dZexpanded = np.matmul(dAdZ,dC_dAexpanded)

		return dC_dZexpanded.reshape(dCdA.shape[1],-1).T

class LeakyReLU(BaseActivation):
	
	def _forwards(self,X:np.ndarray):
		self.input = X.copy()
		# The Leaky ReLu function has a small positive slope in its negative area, enabling it to process zero or negative values.
		self.output = X
		self.output[self.output <= 0] = self.alpha * self.output[self.output <= 0]
		return self.output

	def _backwards(self,dCdA:np.ndarray):
		# Init dAdZ as square array representing all connections between input and output nodes
		dAdZ = np.zeros(shape=(self.output.shape[1],self.output.shape[0],self.prev_layer.output.shape[0]))	# TODO: Will need varifying for Conv Activation.

		ix,iy = np.diag_indices_from(dAdZ[0,:,:])
		dAdZ[:,iy,ix] = ( (self.input > 0).astype(int) + ((self.input < 0).astype(int) * self.alpha ) ).T

		dC_dAexpanded = dCdA.T.reshape((dCdA.T.shape[0],-1,1))
		dC_dZexpanded = np.matmul(dAdZ,dC_dAexpanded)

		return dC_dZexpanded.reshape(dCdA.shape[1],-1).T
