import numpy as np
from .base import BaseActivation

class Softmax(BaseActivation):
	
	def _forwards(self,X:np.ndarray):
		self.input = X.copy()
		assert self.prev_layer.LAYER_TYPE == 'FC', 'Softmax activation function is not supported for non-FC inputs.'
		# Softmax is a special activation function used for output neurons. It normalizes outputs for each class between 0 and 1, and returns the probability that the input belongs to a specific class.
		exp = np.exp(X - np.max(X,axis=0))	# Normalises by max value - provides "numerical stability"
		self.output = exp / np.sum(exp,axis=0)
		return self.output

	def _backwards(self,dCdA:np.ndarray):
		# Vectorised implementation from https://stackoverflow.com/questions/59286911/vectorized-softmax-gradient
		# NOTE: Transpose is required to create the square matrices of each set of node values.
		outputT = self.output.T
		diag_matrices = outputT.reshape(outputT.shape[0],-1,1) * np.diag(np.ones(outputT.shape[1]))	# Diagonal Matrices
		outer_product = np.matmul(outputT.reshape(outputT.shape[0],-1,1), outputT.reshape(outputT.shape[0],1,-1))	# Outer product
		Jsm = diag_matrices - outer_product
		dAdZ = Jsm	# NOTE: Even though this equation uses softmax transpose at start, the output does not require transposing because the softmax derivative is symmetrical along diagonal.

		dC_dAexpanded = dCdA.T.reshape((dCdA.T.shape[0],-1,1))
		dC_dZexpanded = np.matmul(dAdZ,dC_dAexpanded)

		return dC_dZexpanded.reshape(dCdA.shape[1],-1).T
