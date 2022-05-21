import numpy as np
from .base import BaseActivation

class Tanh(BaseActivation):
	ALIAS = "tanh"
	
	def _forwards(self,X:np.ndarray):
		self.input = X.copy()
		# The TanH function is zero-centered making it easier to model inputs that are strongly negative strongly positive or neutral.
		self.output = ( np.exp(X) - np.exp(-X) ) / ( np.exp(X) + np.exp(-X) )
		return self.output

	def _backwards(self,dCdA:np.ndarray):
		dAdZ = np.diag((1 - np.square( self.output )).flatten())

		dC_dAexpanded = dCdA.T.reshape((dCdA.T.shape[0],-1,1))
		dC_dZexpanded = np.matmul(dAdZ,dC_dAexpanded)

		return dC_dZexpanded.reshape(dCdA.shape[1],-1).T
