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

	def prepare_layer(self):
		if self.prev_layer == None:	# This means this is the first layer in the structure, so 'input' is the only thing before.
			assert self.INPUT_SHAPE is not None, 'ERROR: Must define input shape for first layer.'
		else:
			self.INPUT_SHAPE = self.prev_layer.OUTPUT_SHAPE		# (channels, rows, cols)

		assert len(self.INPUT_SHAPE) == 3, 'Invalid INPUT_SHAPE'

		# # Convert 2D input to 3D.
		# if len(self.INPUT_SHAPE) == 2:
		# 	self.INPUT_SHAPE = tuple([1]) + self.INPUT_SHAPE	

		NUM_INPUT_ROWS = self.INPUT_SHAPE[-2]
		NUM_INPUT_COLS = self.INPUT_SHAPE[-1]

		# Initiate params
		self.filters = CNNParam(
			utils.array_init(shape=(self.NUM_FILTERS,self.INPUT_SHAPE[0],self.FILT_SHAPE[0],self.FILT_SHAPE[1]),method=self.INITIATION_METHOD,seed=self.RANDOM_SEED),
			trainable=True
		)
		self.bias = CNNParam(
			np.zeros(shape=(self.NUM_FILTERS,1)),
			trainable=True
		)

		# Need to account for padding.
		if self.PAD_TYPE != None:
			if self.PAD_TYPE == 'same':
				pad_cols_needed = max((NUM_INPUT_COLS - 1) * self.STRIDE + self.FILT_SHAPE[1] - NUM_INPUT_COLS, 0)
				pad_rows_needed = max((NUM_INPUT_ROWS - 1) * self.STRIDE + self.FILT_SHAPE[0] - NUM_INPUT_ROWS, 0)
			elif self.PAD_TYPE == 'valid':
				# TensoFlow definition of this is "no padding". The input is just processed as-is.
				pad_rows_needed = pad_cols_needed = 0
			elif self.PAD_TYPE == 'include':
				# Here we will implement the padding method to avoid input data being excluded/ missed by the convolution.
				# - This happens when, (I_dim - F_dim) % stride != 0
				pad_rows_needed = ((NUM_INPUT_ROWS - self.FILT_SHAPE[0]) % self.STRIDE)
				pad_cols_needed = ((NUM_INPUT_COLS - self.FILT_SHAPE[1]) % self.STRIDE)

			self.COL_LEFT_PAD = pad_cols_needed // 2	# // Floor division
			self.COL_RIGHT_PAD = math.ceil(pad_cols_needed / 2)
			self.ROW_UP_PAD = pad_rows_needed // 2	# // Floor division
			self.ROW_DOWN_PAD = math.ceil(pad_rows_needed / 2)
		else:
			self.COL_LEFT_PAD = self.COL_RIGHT_PAD = self.ROW_UP_PAD = self.ROW_DOWN_PAD = self.PADDING

		col_out = int((NUM_INPUT_COLS + (self.COL_LEFT_PAD + self.COL_RIGHT_PAD) - self.FILT_SHAPE[1]) / self.STRIDE) + 1
		row_out = int((NUM_INPUT_ROWS + (self.ROW_DOWN_PAD + self.ROW_UP_PAD) - self.FILT_SHAPE[0]) / self.STRIDE) + 1

		self.OUTPUT_SHAPE = (self.NUM_FILTERS,row_out,col_out)
		
		# self.output = np.zeros(shape=(self.NUM_FILTERS,row_out,col_out))	# Output initiated.
		if self.PAD_TYPE == 'same':
			assert self.OUTPUT_SHAPE[-2:] == self.INPUT_SHAPE[-2:], f'"SAME" padding chosen however last two dimensions of input and output shapes do not match; {self.INPUT_SHAPE} and {self.OUTPUT_SHAPE} respectively.'	# Channels may differ.


	def _forwards(self,_input):
		assert _input.ndim == 4 and _input.shape[1:] == self.INPUT_SHAPE, f'Input shape, {_input.shape[1:]}, expected to be, {self.INPUT_SHAPE} for each example (observation).'
		self.input = _input
		batch_size = _input.shape[0]

		# Apply the padding to the input.
		self.padded_input = np.pad(self.input,[(0,0),(0,0),(self.ROW_UP_PAD,self.ROW_DOWN_PAD),(self.COL_LEFT_PAD,self.COL_RIGHT_PAD)],'constant',constant_values=(0,0))

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

	def _backwards(self,cost_gradient):	
		assert cost_gradient.shape == self.output.shape, f'cost_gradient shape {cost_gradient.shape} does not match layer output shape {self.output.shape}.'
		if self.TRACK_HISTORY: self._track_metrics(cost_gradient=cost_gradient)
		_,_, c_rows, c_cols = cost_gradient.shape
		dilation_idx_row = np.arange(c_rows-1) + 1	# Intiatial indices for insertion of zeros
		dilation_idx_col = np.arange(c_cols-1) + 1	# Intiatial indices for insertion of zeros

		cost_gradient_dilated = cost_gradient.copy()
		if self.STRIDE != 1:
			for n in range(1,self.STRIDE):	# the n multiplier is to increment the indices in the non-uniform manner required.
				cost_gradient_dilated = np.insert(
					np.insert( cost_gradient_dilated, dilation_idx_row * n, 0, axis=2 ),
					dilation_idx_col * n, 0, axis=3)
		# print(f'cost_gradient shape: {cost_gradient.shape} | cost_gradient_dilated shape: {cost_gradient_dilated.shape}')

		batch_size, channels, _, _ = self.padded_input.shape

		# Account for filter not shifting over input an integer number of times with given stride.
		pxls_excl_x = (self.padded_input.shape[3] - self.FILT_SHAPE[1]) % self.STRIDE	# pixels excluded in x direction (cols)
		pxls_excl_y = (self.padded_input.shape[2] - self.FILT_SHAPE[0]) % self.STRIDE	# pixels excluded in y direction (rows)
		# print('PIXELS EXCLUDED:',pxls_excl_x,pxls_excl_y)

		# Find cost gradient wrt previous output and filters.
		rotated_filters = np.rot90( self.filters, k=2, axes=(1,2) )	# rotate 2x90 degs, rotating in direction of rows to columns.
		dCdX_pad = np.zeros(shape=self.padded_input.shape)
		if self.VECTORISED:
			dCdF = np.transpose(Conv2D.convolve_vectorised(np.transpose(self.padded_input[:,:, :self.padded_input.shape[2] - pxls_excl_y, :self.padded_input.shape[3] - pxls_excl_x],axes=(1,0,2,3)),np.transpose(cost_gradient_dilated,axes=(1,0,2,3)),stride=1),axes=(1,0,2,3))
			dCdX_pad[:,:, :dCdX_pad.shape[2] - pxls_excl_y, :dCdX_pad.shape[3] - pxls_excl_x] = Conv2D.convolve_vectorised(cost_gradient_dilated,np.transpose(rotated_filters,axes=(1,0,2,3)),stride=1,full_convolve=True)
		else:
			dCdF = np.zeros(shape=self.filters.shape)
			for i in range(batch_size):
				for filt_index in range(self.NUM_FILTERS):
					for channel_index in range(channels):
						dCdF[filt_index, channel_index] += Conv2D.convolve( self.padded_input[i,channel_index, :self.padded_input.shape[2] - pxls_excl_y, :self.padded_input.shape[3] - pxls_excl_x], cost_gradient_dilated[i,filt_index], stride=1 )

						dCdX_pad[i,channel_index, :dCdX_pad.shape[2] - pxls_excl_y, :dCdX_pad.shape[3] - pxls_excl_x] += Conv2D.convolve( cost_gradient_dilated[i,filt_index], rotated_filters[filt_index,channel_index], stride=1, full_convolve=True )
			
		# dCdF = dCdF[:,:, : dCdF.shape[2] - pxls_excl_y, : dCdF.shape[3] - pxls_excl_x]	# Remove the values from right and bottom of array (this is where the excluded pixels will be).
		
		# ADJUST THE FILTERS
		assert dCdF.shape == self.params['filters']['values'].shape, f'dCdF shape {dCdF.shape} does not match filters shape {self.params["filters"]["values"].shape}.'
		self.filters.gradient = dCdF
		if self.filters.trainable:
			self.filters = self.model.OPTIMISER.update_param(self.filters)

		# ADJUST THE BIAS
		dCdB = 1 * cost_gradient.sum(axis=(0,2,3)).reshape(self.bias.shape)
		assert dCdB.shape == self.bias.shape, f'dCdB shape {dCdB.shape} does not match bias shape {self.bias.shape}.'
		self.bias.gradient = dCdB
		if self.bias.trainable:
			self.bias = self.model.OPTIMISER.update_param(self.bias)

		# Remove padding that was added to the input array.
		dCdX = dCdX_pad[ :, : , self.ROW_UP_PAD : dCdX_pad.shape[-2] - self.ROW_DOWN_PAD , self.COL_LEFT_PAD : dCdX_pad.shape[-1] - self.COL_RIGHT_PAD ]
		assert dCdX.shape == self.input.shape, f'dCdX shape [{dCdX.shape}] does not match layer input shape [{self.input.shape}].'

		return dCdX

	@staticmethod
	def convolve(A, B, stride,full_convolve=False):
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
	def convolve_vectorised(X,K, stride, full_convolve=False):
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
