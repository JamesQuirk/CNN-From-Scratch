import numpy as np
from .layer import Layer
import math
from cnn import utils

class Pool(Layer):
	def __init__(self,filt_shape: tuple or int,stride: int,pool_type: str='max',padding: int=0,pad_type: str=None,input_shape=None,vectorised=True,track_history=True):
		'''
		- filt_shape (int/tuple): Number of rows and columns of each filter. INT if rows == cols. TUPLE if rows != cols.
		- stride (int): Size of steps to take when shifting the filter. (Currently stride_x = stride_y).
		- pool_type (str): Pooling method to be applied. Options = max, min, mean.
		- padding (int): Width of zero-padding to apply on each side of the array. Only applied if pad_type is None.
		- pad_type (str): Options: same (output shape is same as input shape), valid (equal to padding=0), include (padding added evenly on all sides of the array to allow the filter to shift over the input an integer number of times - avoid excluding input data).
		- input_shape (tuple): Input shape of a single example (observation). Expected (channels, rows, cols)
		'''
		super().__init__()

		self.LAYER_TYPE = self.__class__.__name__
		self.TRAINABLE = False
		if type(filt_shape) == tuple:
			assert len(filt_shape) == 2, 'Expected 2 dimensional tuple in form: (rows,cols)'
			self.FILT_SHAPE = filt_shape	# 2D tuple describing num rows and cols
		elif type(filt_shape) == int:
			self.FILT_SHAPE = (filt_shape,filt_shape)
		self.STRIDE = stride
		self.POOL_TYPE = pool_type.lower()
		assert self.POOL_TYPE in ('max','mean','min')
		self.PADDING = padding
		self.PAD_TYPE = None if pad_type is None else pad_type.lower()
		if input_shape is not None:
			assert len(input_shape) == 3, 'input_shape must be of length 3: (num_channels, num_rows, num_columns)'
		self.INPUT_SHAPE = input_shape
		self.VECTORISED = vectorised
		self.TRACK_HISTORY = track_history

		self.NUM_PARAMS = 0

	def prepare_layer(self) -> np.ndarray:
		""" This needs to be done after the input has been identified - currently happens when train() is called. """
		if self.prev_layer == None:	# This means this is the first layer in the structure, so 'input' is the only thing before.
			assert self.INPUT_SHAPE is not None, 'ERROR: Must define input shape for first layer.'
		else:
			self.INPUT_SHAPE = self.prev_layer.OUTPUT_SHAPE		# (channels, rows, cols)

		assert len(self.INPUT_SHAPE) == 3, 'Invalid INPUT_SHAPE'

		self.COL_LEFT_PAD, self.COL_RIGHT_PAD, self.ROW_UP_PAD, self.ROW_DOWN_PAD = utils.array.determine_padding(
			self.PAD_TYPE, self.PADDING, self.INPUT_SHAPE, self.FILT_SHAPE, self.STRIDE
		)
		col_out = int((self.INPUT_SHAPE[2] + (self.COL_LEFT_PAD + self.COL_RIGHT_PAD) - self.FILT_SHAPE[1]) / self.STRIDE) + 1
		row_out = int((self.INPUT_SHAPE[1] + (self.ROW_DOWN_PAD + self.ROW_UP_PAD) - self.FILT_SHAPE[0]) / self.STRIDE) + 1

		self.OUTPUT_SHAPE = (self.INPUT_SHAPE[0],row_out,col_out)
		if self.PAD_TYPE == 'same':
			assert self.OUTPUT_SHAPE == self.INPUT_SHAPE	# Channels may differ.

	def _forwards(self,_input: np.ndarray) -> np.ndarray:
		assert _input.ndim == 4 and _input.shape[1:] == self.INPUT_SHAPE, f'Input shape, {_input.shape[1:]}, expected to be, {self.INPUT_SHAPE} for each example (observation).'
		self.input = _input

		# Apply the padding to the input.
		self.padded_input = np.pad(self.input,[(0,0),(0,0),(self.ROW_UP_PAD,self.ROW_DOWN_PAD),(self.COL_LEFT_PAD,self.COL_RIGHT_PAD)],'constant',constant_values=(0,0))

		if self.VECTORISED:
			self.Xsliced = np.zeros((self.padded_input.shape[0],self.padded_input.shape[1],np.prod(self.FILT_SHAPE),np.prod(self.OUTPUT_SHAPE[-2:])))
			col_index = 0
			for vstart in range(0,self.padded_input.shape[-2] - self.FILT_SHAPE[0] + 1, self.STRIDE):
				for hstart in range(0, self.padded_input.shape[-1] - self.FILT_SHAPE[1] + 1, self.STRIDE):
					self.Xsliced[:,:,:,col_index] = np.transpose(self.padded_input[:,:,vstart:vstart+self.FILT_SHAPE[0],hstart:hstart+self.FILT_SHAPE[1]],axes=(0,1,3,2)).reshape((*self.padded_input.shape[:2],np.prod(self.FILT_SHAPE)))
					col_index += 1
			if self.POOL_TYPE == 'max':
				X_flat_pooled = np.max(self.Xsliced, axis=2)
			elif self.POOL_TYPE == 'mean':
				X_flat_pooled = np.mean(self.Xsliced, axis=2)
			elif self.POOL_TYPE == 'min':
				X_flat_pooled = np.min(self.Xsliced,axis=2)
			self.output = X_flat_pooled.reshape((self.padded_input.shape[0],*self.OUTPUT_SHAPE))
		else:
			self.output = np.zeros(shape=(self.input.shape[0],*self.OUTPUT_SHAPE))
			batch_size, channels, proc_rows, proc_cols = self.padded_input.shape
			for i in range(batch_size):
				# Shift 'Filter Window' over the image and perform the downsampling
				curr_y = out_y = 0
				while curr_y <= proc_rows - self.FILT_SHAPE[0]:
					curr_x = out_x = 0
					while curr_x <= proc_cols - self.FILT_SHAPE[1]:
						for channel_index in range(channels):
							sub_arr = self.padded_input[i, channel_index, curr_y : curr_y + self.FILT_SHAPE[0], curr_x : curr_x + self.FILT_SHAPE[1] ]
							if self.POOL_TYPE == 'max':
								self.output[i,channel_index, out_y, out_x] = np.max( sub_arr )
							elif self.POOL_TYPE == 'min':
								self.output[i,channel_index, out_y, out_x] = np.min( sub_arr )
							elif self.POOL_TYPE == 'mean':
								self.output[i,channel_index, out_y, out_x] = np.mean( sub_arr )

						curr_x += self.STRIDE
						out_x += 1
					curr_y += self.STRIDE
					out_y += 1

		assert len(self.output.shape) == 4 and self.output.shape[1:] == self.OUTPUT_SHAPE, f'Output shape, {self.output.shape[1:]}, not as expected, {self.OUTPUT_SHAPE}'
		if self.TRACK_HISTORY: self._track_metrics(output=self.output)
		return self.output

	def _backwards(self,cost_gradient: np.ndarray) -> np.ndarray:
		'''
		Backprop in pooling layer:
		- nothing to be updated as there are no weights in this layer.
		- just need to propogate the cost gradient backwards.

		Cost gradient received as an array in the same shape as this layer's output. Need to 'fill in the blanks' as this layer removed data in the forwards pass.

		If Pooling is MAX:
		- The responsibility of the whole cost gradient associated with the given region of the input is with the node with the maximum value.
		- All others will have cost gradient of 0.

		If Pooling is MEAN:
		- The responsibility will be split between the nodes; weighted by the proportion of each value to the total for the region.
		'''
		assert cost_gradient.shape == self.output.shape
		if self.TRACK_HISTORY: self._track_metrics(cost_gradient=cost_gradient)
		# Initiate to input shape.
		dC_dIpad = np.zeros_like(self.padded_input)

		batch_size, channels, padded_rows, padded_cols = dC_dIpad.shape

		if self.VECTORISED:
			if self.POOL_TYPE == 'max':
				distribution_arr = (np.max(self.Xsliced,axis=2,keepdims=True) == self.Xsliced).astype(int)
			elif self.POOL_TYPE == 'min':
				distribution_arr = (np.min(self.Xsliced,axis=2,keepdims=True) == self.Xsliced).astype(int)
			elif self.POOL_TYPE == 'mean':
				distribution_arr = np.ones_like(self.Xsliced)
			cg_flat = cost_gradient.reshape((*self.Xsliced.shape[:2],1,self.Xsliced.shape[-1])) * distribution_arr
			col_index = 0
			for vstart in range(0,self.padded_input.shape[-2] - self.FILT_SHAPE[0] + 1, self.STRIDE):
				for hstart in range(0, self.padded_input.shape[-1] - self.FILT_SHAPE[1] + 1, self.STRIDE):
					dC_dIpad[:,:,vstart:vstart+self.FILT_SHAPE[0],hstart:hstart+self.FILT_SHAPE[1]] += np.transpose(cg_flat[:,:,:,col_index].reshape((*self.padded_input.shape[:2],*self.FILT_SHAPE[::-1])),axes=(0,1,3,2))
					col_index += 1
		else:
			# Step over the array similarly to the forwards pass and compute the expanded cost gradients.
			for i in range(batch_size):
				curr_y = cost_y = 0
				while curr_y <= padded_rows - self.FILT_SHAPE[0]:
					curr_x = cost_x = 0
					while curr_x <= padded_cols - self.FILT_SHAPE[1]:
						for channel_index in range(channels):
							sub_arr = self.padded_input[i, channel_index, curr_y : curr_y + self.FILT_SHAPE[0], curr_x : curr_x + self.FILT_SHAPE[1] ]
							cost_val = cost_gradient[i, channel_index,cost_y,cost_x]
							if self.POOL_TYPE == 'max':
								# Set value of node that corresponds with the max value node of the input to the cost gradient value at (cost_y,cost_x)
								max_node_y, max_node_x = np.array( np.unravel_index( np.argmax( sub_arr ), sub_arr.shape ) ) + np.array([curr_y, curr_x])	# addition of curr_y & curr_x is to get position in padded_input array (not just local sub_arr).

								dC_dIpad[i, channel_index, max_node_y, max_node_x] += cost_val
							elif self.POOL_TYPE == 'min':
								# Set value of node that corresponds with the min value node of the input to the cost gradient value at (cost_y,cost_x)
								min_node_y, min_node_x = np.array( np.unravel_index( np.argmin( sub_arr ), sub_arr.shape ) ) + np.array([curr_y, curr_x])	# addition of curr_y & curr_x is to get position in padded_input array (not just local sub_arr).

								dC_dIpad[i, channel_index, min_node_y, min_node_x] += cost_val
							elif self.POOL_TYPE == 'mean':
								sub_arr_props = sub_arr / sub_arr.sum()

								dC_dIpad[i, channel_index, curr_y : curr_y + self.FILT_SHAPE[0], curr_x : curr_x + self.FILT_SHAPE[1] ] += sub_arr_props * cost_val

						curr_x += self.STRIDE
						cost_x += 1
					curr_y += self.STRIDE
					cost_y += 1

		# Remove padding that was added to the input array.
		dC_dI = dC_dIpad[ :, : , self.ROW_UP_PAD : dC_dIpad.shape[-2] - self.ROW_DOWN_PAD , self.COL_LEFT_PAD : dC_dIpad.shape[-1] - self.COL_RIGHT_PAD ]
		assert dC_dI.shape == self.input.shape, f'dC/dI shape [{dC_dI.shape}] does not match layer input shape [{self.input.shape}].'
		return dC_dI
