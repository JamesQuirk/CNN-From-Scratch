import numpy as np
from cnn import utils
from cnn.params import CNNParam
from .layer import Layer
import math

class Conv2D(Layer):
	def __init__(self,filt_shape: tuple or int,num_filters: int=5,stride: int=1,padding: int=0,pad_type: str=None,random_seed=42,initiation_method=None,input_shape=None,vectorised=True,track_history=True):
		""" 
		- filt_shape (int/tuple): Number of rows and columns of each filter. INT if rows == cols. TUPLE if rows != cols.
		- num_filters (int): Number of filters to be used.
		- stride (int): Size of steps to take when shifting the filter. (Currently stride_x = stride_y).
		- padding (int): Width of zero-padding to apply on each side of the array. Only applied if pad_type is None.
		- pad_type (str): Options: same (output shape is same as input shape), valid (equal to padding=0), include (padding added evenly on all sides of the array to allow the filter to shift over the input an integer number of times - avoid excluding input data).
		- random_seed (int): The seed provided to numpy before initiating the filters and biases. random_seed=None will result in no seed being provided meaning numpy will generate it dynamically each time.
		- initiation_method (str): This is the method used to initiate the weights and biases. Options: "kaiming", "xavier" or None. Default is none - this simply takes random numbers from standard normal distribution with no scaling.
		- input_shape (tuple): The input shape of single example (observation).
		"""
		assert num_filters > 0, 'Cannot use less than 1 filter in Conv Layer.'
		super().__init__()

		self.LAYER_TYPE = self.__class__.__name__
		self.TRAINABLE = True
		if type(filt_shape) == tuple:
			assert len(filt_shape) == 2, 'Expected 2 dimensional tuple in form: (rows,cols)'
			self.FILT_SHAPE = filt_shape	# 2D tuple describing num rows and cols
		elif type(filt_shape) == int:
			self.FILT_SHAPE = (filt_shape,filt_shape)
		self.NUM_FILTERS = num_filters
		self.STRIDE = stride
		self.PADDING = padding
		self.PAD_TYPE = None if pad_type is None else pad_type.lower()
		self.RANDOM_SEED = random_seed
		self.INITIATION_METHOD = None if initiation_method is None else initiation_method.lower()
		if input_shape is not None:
			assert len(input_shape) == 3, 'input_shape must be of length 3: (num_channels, num_rows, num_columns)'
		self.INPUT_SHAPE = input_shape
		self.VECTORISED = vectorised
		self.TRACK_HISTORY = track_history

	def prepare_layer(self) -> None:
		if self.prev_layer == None:	# This means this is the first layer in the structure, so 'input' is the only thing before.
			assert self.INPUT_SHAPE is not None, 'ERROR: Must define input shape for first layer.'
		else:
			self.INPUT_SHAPE = self.prev_layer.OUTPUT_SHAPE		# (channels, rows, cols)

		assert len(self.INPUT_SHAPE) == 3, 'Invalid INPUT_SHAPE'

		# Initiate params
		self.filters = CNNParam(
			utils.array.array_init(shape=(self.NUM_FILTERS,self.INPUT_SHAPE[0],self.FILT_SHAPE[0],self.FILT_SHAPE[1]),method=self.INITIATION_METHOD,seed=self.RANDOM_SEED),
			trainable=True
		)
		self.bias = CNNParam(
			np.zeros(shape=(self.NUM_FILTERS,1)),
			trainable=True
		)

		# Need to account for padding.
		self._COL_LEFT_PAD, self._COL_RIGHT_PAD, self._ROW_UP_PAD, self._ROW_DOWN_PAD = utils.array.determine_padding(
			self.PAD_TYPE, self.PADDING, self.INPUT_SHAPE, self.FILT_SHAPE, self.STRIDE
		)

		col_out = int((self.INPUT_SHAPE[2] + (self._COL_LEFT_PAD + self._COL_RIGHT_PAD) - self.FILT_SHAPE[1]) / self.STRIDE) + 1
		row_out = int((self.INPUT_SHAPE[1] + (self._ROW_DOWN_PAD + self._ROW_UP_PAD) - self.FILT_SHAPE[0]) / self.STRIDE) + 1

		self.OUTPUT_SHAPE = (self.NUM_FILTERS,row_out,col_out)
		
		# self.output = np.zeros(shape=(self.NUM_FILTERS,row_out,col_out))	# Output initiated.
		if self.PAD_TYPE == 'same':
			assert self.OUTPUT_SHAPE[-2:] == self.INPUT_SHAPE[-2:], f'"SAME" padding chosen however last two dimensions of input and output shapes do not match; {self.INPUT_SHAPE} and {self.OUTPUT_SHAPE} respectively.'	# Channels may differ.


	def _forwards(self,_input: np.ndarray) -> np.ndarray:
		assert _input.ndim == 4 and _input.shape[1:] == self.INPUT_SHAPE, f'Input shape, {_input.shape[1:]}, expected to be, {self.INPUT_SHAPE} for each example (observation).'
		self.input = _input
		batch_size = _input.shape[0]

		# Apply the padding to the input.
		self.padded_input = np.pad(self.input,[(0,0),(0,0),(self._ROW_UP_PAD,self._ROW_DOWN_PAD),(self._COL_LEFT_PAD,self._COL_RIGHT_PAD)],'constant',constant_values=(0,0))

		if self.VECTORISED:
			self.output = Conv2D.convolve_vectorised(self.padded_input,self.filters,self.STRIDE)
			self.output += np.broadcast_to(
				self.bias[:,None,:],	# Insert axis for the array items to be expanded into
				self.output.shape
			)
		else:
			self.output = np.zeros(shape=(batch_size,*self.OUTPUT_SHAPE))
			for i in range(batch_size):
				for filt_index in range(self.NUM_FILTERS):
					filt = self.filters[filt_index]
					filt_channels, _, _ = filt.shape

					for channel_index in range(filt_channels):
						self.output[i,filt_index] += Conv2D.convolve( self.padded_input[i,channel_index], filt[channel_index], self.STRIDE )
					
					self.output[i,filt_index] += self.bias[filt_index]

		if self.TRACK_HISTORY: self._track_metrics(output=self.output)
		return self.output	# NOTE: Output is 4D array of shape: ( BATCH_SIZE, NUM_FILTS, NUM_ROWS, NUM_COLS )

	def _backwards(self,cost_gradient: np.ndarray) -> np.ndarray:	
		assert cost_gradient.shape == self.output.shape, f'cost_gradient shape {cost_gradient.shape} does not match layer output shape {self.output.shape}.'
		if self.TRACK_HISTORY: self._track_metrics(cost_gradient=cost_gradient)

		cost_gradient_dilated = utils.array.dilate(cost_gradient,self.STRIDE-1)

		batch_size, channels, _, _ = self.padded_input.shape

		# Account for filter not shifting over input an integer number of times with given stride. In this case, 
		# the 'effective input is smaller than the actual input.
		pxls_excl_x = (self.padded_input.shape[3] - self.FILT_SHAPE[1]) % self.STRIDE	# pixels excluded in x direction (cols)
		pxls_excl_y = (self.padded_input.shape[2] - self.FILT_SHAPE[0]) % self.STRIDE	# pixels excluded in y direction (rows)

		# Extract effective input
		effective_input = self.padded_input[
			:,	# All data points 
			:,	# All channels
			:self.padded_input.shape[2] - pxls_excl_y, # Only rows up to those excluded in forwards pass
			:self.padded_input.shape[3] - pxls_excl_x	# Only cols up to those excluded in forwards pass
			]

		# Find cost gradient wrt layer input and filters.
		rotated_filters = np.rot90( self.filters, k=2, axes=(2,3) )	# rotate 2x90 degs, rotating in direction of rows to columns.
		if self.VECTORISED:
			# NOTE: convolution function sums across channels; in this case we want to sum across batch data points so we 
			# transpose the arrays to switch the 'channels' with the 'batch' fields. We then need to switch these back for the
			# resultant array.
			dCdF = np.transpose(
				Conv2D.convolve_vectorised(
					np.transpose(
						effective_input,
						axes=(1,0,2,3)
					),
					np.transpose(
						cost_gradient_dilated,
						axes=(1,0,2,3)
					),
					stride=1
				),
				axes=(1,0,2,3))
			# NOTE: Here we need to transpose the filters to allign the channels of the filters with the batched data points in the cost gradient array.
			effective_input_gradient = Conv2D.convolve_vectorised(
				cost_gradient_dilated,
				np.transpose(
					rotated_filters,
					axes=(1,0,2,3)
				),
				stride=1,
				full_convolve=True)
		else:
			dCdF = np.zeros(shape=self.filters.shape)
			effective_input_gradient = np.zeros(shape=effective_input.shape)
			for i in range(batch_size):
				for filt_index in range(self.NUM_FILTERS):
					for channel_index in range(channels):
						dCdF[filt_index, channel_index] += Conv2D.convolve( effective_input[i,channel_index,:,:], cost_gradient_dilated[i,filt_index], stride=1 )
						effective_input_gradient[i,channel_index, :, :] += Conv2D.convolve( cost_gradient_dilated[i,filt_index], rotated_filters[filt_index,channel_index], stride=1, full_convolve=True )
		
		# ADJUST THE FILTERS
		assert dCdF.shape == self.filters.shape, f'dCdF shape {dCdF.shape} does not match filters shape {self.filters.shape}.'
		self.filters.gradient = dCdF
		if self.filters.trainable:
			self.filters = self.model.OPTIMISER.update_param(self.filters)

		# ADJUST THE BIAS
		dCdB = 1 * cost_gradient.sum(axis=(0,2,3)).reshape(self.bias.shape)
		assert dCdB.shape == self.bias.shape, f'dCdB shape {dCdB.shape} does not match bias shape {self.bias.shape}.'
		self.bias.gradient = dCdB
		if self.bias.trainable:
			self.bias = self.model.OPTIMISER.update_param(self.bias)

		# Obtain dCdX, accounting for padding and excluded input values
		dCdX_pad = np.zeros(shape=self.padded_input.shape)
		dCdX_pad[:,:, :dCdX_pad.shape[2] - pxls_excl_y, :dCdX_pad.shape[3] - pxls_excl_x] = effective_input_gradient
		dCdX = dCdX_pad[ :, : , self._ROW_UP_PAD : dCdX_pad.shape[-2] - self._ROW_DOWN_PAD , self._COL_LEFT_PAD : dCdX_pad.shape[-1] - self._COL_RIGHT_PAD ]
		assert dCdX.shape == self.input.shape, f'dCdX shape [{dCdX.shape}] does not match layer input shape [{self.input.shape}].'

		return dCdX

	@staticmethod
	def convolve(A: np.ndarray, B: np.ndarray, stride: int,full_convolve: bool=False) -> np.ndarray:
		""" A and B are 2D arrays. Array B will be convolved over Array A using the stride provided.
			- 'full_convolve' is where the bottom right cell of B starts over the top of the top left cell of A and shifts by stride until the top left cell of B is over the bottom right cell of A. (i.e. A is padded in each dimension by B - 1 in the respective dimension). """
		assert A.ndim == 2
		assert B.ndim == 2
		if full_convolve:
			vertical_pad = B.shape[0] - 1
			horizontal_pad = B.shape[1] - 1
			A = np.pad(A,[(vertical_pad,vertical_pad),(horizontal_pad,horizontal_pad)],'constant')
		arows, acols = A.shape
		brows, bcols = B.shape

		rout = int((arows - brows) / stride) + 1
		cout = int((acols - bcols) / stride) + 1

		output = np.zeros(shape=(rout,cout))

		# start with mask in top left corner
		curr_y = out_y = 0	# 'curr_y' is y position of the top left corner of filt on top of '_input'. 'out_y' is the corresponding y position in the output array.
		while curr_y <= arows - brows:
			curr_x = out_x = 0	# 'curr_x' is x position of the top left corner of filt on top of '_input'. 'out_x' is the corresponding x position in the output array.
			while curr_x <= acols - bcols:
				output[out_y,out_x] += np.sum( A[ curr_y : curr_y + brows, curr_x : curr_x + bcols ] * B)
				curr_x += stride
				out_x += 1

			curr_y += stride
			out_y += 1
		return output

	@staticmethod
	def convolve_vectorised(X: np.ndarray,K: np.ndarray, stride: int, full_convolve: bool=False) -> np.ndarray:
		"""
		X: 4D array of shape: (batch_size,channels,rows,cols)
		K: 4D array of shape: (num_filters,X_channels,rows,cols)

		Speed of this function is inversely proportional to [X_rows - K_rows] * [X_cols - K_cols].
		- Therefore, the larger the difference between size of X compared with K, the longer the function takes to run - this is due to the nested loop.
		"""
		assert X.ndim == 4 and K.ndim == 4, 'X and K should be 4D arrays.'
		assert X.shape[1] == K.shape[1], f'Both X and K should have the same number of channels. X has {X.shape[1]} and K has {K.shape[1]}.'
		X = X.copy()
		K = K.copy()
		
		if full_convolve:
			vertical_pad = K.shape[2] - 1
			horizontal_pad = K.shape[3] - 1
			X = np.pad(X,[(0,0),(0,0),(vertical_pad,vertical_pad),(horizontal_pad,horizontal_pad)],'constant',constant_values=0)

		# Flatten last 2 dimensions of K so that it becomes a 3D array with shape (num filts, K_rows * K_cols * X channels)
		Kflat = np.transpose(K,axes=(0,1,3,2)).reshape((K.shape[0],np.prod(K.shape[1:])))

		# Extract each slice of X for the conv operation and place into columns of Xsliced
		fmap_rows = int((X.shape[2] - K.shape[2]) / stride) + 1
		fmap_cols = int((X.shape[3] - K.shape[3]) / stride) + 1
		Xsliced = np.zeros((X.shape[0],np.prod(K.shape[1:]),fmap_rows*fmap_cols))
		col_index = 0
		for vstart in range(0,X.shape[2] - K.shape[2] + 1,stride):
			for hstart in range(0,X.shape[3] - K.shape[3] + 1,stride):	# NOTE: This double for loop can become slow when X inner shape is significantly greater than K inner shape (rows,cols)
				Xsliced[:,:,col_index] = np.transpose(X[:,:,vstart:vstart+K.shape[2],hstart:hstart+K.shape[3]],axes=(0,1,3,2)).reshape((X.shape[0],np.prod(K.shape[1:])))
				col_index += 1
		Fmap_flat = np.matmul(Kflat, Xsliced)	# (batch size, num filts, fmap_rows * fmap_cols)
		# Transform Fmap_flat to (batch size, num filts, fmap_rows, fmap_cols)
		return Fmap_flat.reshape((X.shape[0],K.shape[0], fmap_rows,fmap_cols))

	@property
	def filters(self):
		return self._filters

	@filters.setter
	def filters(self,value):
		self._filters = CNNParam(value)

	@property
	def bias(self):
		return self._bias

	@bias.setter
	def bias(self,value):
		self._bias = CNNParam(value)
