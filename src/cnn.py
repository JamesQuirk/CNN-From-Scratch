'''
This is the main class file for the Convolutional Neural Network

CNN Flow:
	Input -> Conv. -> Pooling [-> Conv. -> Pooling] -> Flatten -> Fully Connected layer -> Output

Array indexing convention: (rows,columns) <- consistent with numpy.
'''

# IMPORTS
import numpy as np
import math
import pickle
from datetime import datetime as dt

# CLASS
class CNN():
	"""
	This is the top level class. It contains sub-classes for each of the layers that are to be included in the model.
	"""

	def __init__(self,input_shape: tuple,learning_rate=0.1,cost_fn='mse'):
		assert len(input_shape) == 3, 'input_shape must be of length 3: (num_channels, num_rows, num_columns)'

		self.is_prepared = False

		self.INPUT_SHAPE = input_shape	# tuple to contain input shape
		self.LEARNING_RATE = learning_rate
		self.cost_fn = cost_fn

		self.structure = []	# defines order of model (list of layer objects) - EXCLUDES INPUT DATA
		self.num_layers = {'total':0,'CONV':0,'POOL':0,'FLATTEN':0,'FC':0,'ACTIVATION':0}	# dict for counting number of each layer type
		self.cost_history = []

	def add_layer(self,layer):
		if layer.LAYER_TYPE == 'ACTIVATION' and self.structure[-1].LAYER_TYPE == 'ACTIVATION':
			print('-- WARNING:: Two Activation Layers in subsequent positions in the model.')
			if layer.FUNCTION == self.structure[-1].FUNCTION:
				print('--- INFO:: Both Activation Layers are the same, skipping creation of second layer.')
				return

		layer.model = self

		if len(self.structure) > 0:
			if layer.LAYER_TYPE == 'FC' and self.structure[-1].LAYER_TYPE not in ('FLATTEN','FC','ACTIVATION'):
				# If no Flatten layer added before adding first FC layer, one will be added automatically.
				self.add_layer(CNN.Flatten_Layer())

		self.structure.append(layer)
		self.num_layers[layer.LAYER_TYPE] += 1
		self.num_layers['total'] += 1

		if layer.LAYER_TYPE == 'FC':
			# Create the Activation Layer (transparent to user).
			self.add_layer(
				CNN.Activation(function=layer.ACTIVATION)
			)

	def remove_layer(self,index):
		self.structure.pop(index)
		if self.is_prepared:
			print('-- INFO:: Model preparation will now need to be re-done.')
			self.prepare_model()
			
	def get_model_details(self):
		details = []
		for layer in self.structure:
			details.append(layer.define_details())

		return details
		
	def prepare_model(self):
		""" Called once final layer is added, each layer can now iniate its weights and biases. """
		print('Preparing model...')
		if self.num_layers['total'] > 1:
			for index in range(self.num_layers['total']):
				curr_layer = self.structure[index]
				if index != len(self.structure) - 1:
					next_layer = self.structure[index + 1]
				else:
					next_layer = None

				curr_layer.next_layer = next_layer
				if next_layer is not None:
					next_layer.prev_layer = curr_layer

				curr_layer.model_structure_index = index

				print(f'Preparing Layer:: Type = {curr_layer.LAYER_TYPE} | Structure index = {curr_layer.model_structure_index}')
				curr_layer.prepare_layer()
				print('--> Expected output shape:',curr_layer.output.shape)
		
		self.is_prepared = True
		print(f'Model Prepared: {self.is_prepared}')

	@staticmethod
	def one_hot_encode(array,num_cats,axis=None):
		'''
		Perform one-hot encoding on the category labels.

		- array: is a 2D np.ndarray
		- num_cats: number of categories that the model is to be trained on.
		- axis: the axis of array that holds the category label value. If axis=None, then this is inferred as the axis with the smallest size.
		'''
		assert type(array) in (np.ndarray,list)
		array = np.array(array)
		assert array.ndim == 2
		if axis is None:
			axis = np.argmin(array.shape)
		else:
			assert axis in (0,1)
		assert array.shape[axis] == 1

		N = array.shape[1 - axis]
		array = array.reshape((1,N))
		
		return np.eye(num_cats)[array][0]	# Returns in the shape (N,num_cats)

	def train(self,Xs,ys,epochs,batch_size=None,shuffle=False,random_seed=42):
		'''
		Should take array of inputs and array of labels of the same length.

		[For each epoch] For each input, propogate forwards and backwards.

		ARGS:
		- Xs (np.ndarray or list): (N,ch,rows,cols). Where N is number of examples, ch is number of channels.
		- ys (np.ndarray or list): (N,num_categories). e.g. ex1 label = [0,0,0,1,0] for 5 categories (one-hot encoded).
		- epochs (int): Number of iterations over the data set.
		- batch_size (int): Maximum number of examples in each batch. batch_size=None is equal to batch_size=N.
		- shuffle (bool): Determines whether the data set is shuffled before training.
		- random_seed (int): The seed provided to numpy before performing the shuffling. random_seed=None will result in no seed being provided meaning numpy will generate it dynamically each time.
		'''
		train_start = dt.now()
		if not self.is_prepared:
			self.prepare_model()

		Xs, ys = np.array(Xs), np.array(ys)	# Convert data to numpy arrays in case not already.

		print('Xs shape:', Xs.shape,'ys shape:', ys.shape)

		# Check shapes and orientation are as expected
		assert Xs.shape[0] == ys.shape[0], 'Dimension of input data and labels does not match.'
		assert ys.shape[-1] == self.structure[-1].NUM_NODES, 'Invalid shape for labels. Should be (N,num_categories)'	# NOTE: This assumes last layer in model is FC_Layer.

		N = Xs.shape[0]	# Total number of examples in Xs

		if batch_size is None:
			num_batches = 1
			self.batch_size = N
		else:
			assert int(batch_size) == batch_size, 'An integer value must be supplied for argument "batch_size"'
			self.batch_size = batch_size
			num_batches = math.ceil( N / batch_size )

		if shuffle:
			if random_seed is not None:
				np.random.seed(random_seed)
			permutation = np.random.permutation( N )

			Xs = Xs[permutation]
			ys = ys[permutation]

		# Forwards pass...
		for epoch_ind in range(epochs):
			print(f'------ EPOCH: {epoch_ind + 1} ------')
			for batch_ind in range(num_batches):
				ind_lower = batch_ind * self.batch_size	# Lower bound of index range
				ind_upper = batch_ind * self.batch_size + self.batch_size	# Upper bound of index range
				if ind_upper > N - 1 and N > 1:
					ind_upper = N - 1

				batch_Xs = Xs[ ind_lower : ind_upper ]
				batch_ys = ys[ ind_lower : ind_upper ]

				cost = 0
				cost_gradient = 0

				for ex_ind , X in enumerate(batch_Xs):	# For each example (observation)
					print(f'Epoch: {epoch_ind+1} | Batch: {batch_ind+1} of {math.ceil(N/batch_size)} | Example: {ex_ind+1 + batch_ind*batch_size}')
					for layer in self.structure:
						# print(f'BEFORE LAYER: {layer.LAYER_TYPE} [{X.shape if isinstance(X, np.ndarray) else None}]') if layer.LAYER_TYPE == 'CONV' else None

						X = layer._forwards(X)

						# print(f'AFTER LAYER: {layer.LAYER_TYPE} [{X.shape if isinstance(X, np.ndarray) else None}]') if layer.LAYER_TYPE == 'CONV' else None

					cost += self.cost(X, batch_ys[ex_ind])
					cost_gradient += self.cost(X, batch_ys[ex_ind],derivative=True)	# partial diff of cost w.r.t. output of the final layer

				print(f'-- Epoch index: {epoch_ind} | Batch index: {batch_ind} | Cost: {cost}')
				self.cost_history.append(cost)

				# Backpropagate the cost
				for layer in self.structure[::-1]:
					# print(f'BEFORE LAYER: {layer.LAYER_TYPE} [{cost_gradient.shape if isinstance(cost_gradient, np.ndarray) else None}]') if layer.LAYER_TYPE == 'CONV' else None
					
					cost_gradient = layer._backwards(cost_gradient)

					# print(f'AFTER LAYER: {layer.LAYER_TYPE} [{cost_gradient.shape if isinstance(cost_gradient, np.ndarray) else None}]') if layer.LAYER_TYPE == 'CONV' else None
		return dt.now(), dt.now() - train_start	# returns training finish time and duration.

	def cost(self,prediction,label,derivative=False):
		'''
		Cost function to provide measure of model 'correctness'. returns scalar cost value.
		'''
		label = label.reshape((max(label.shape),1))	# reshape label to vertical array to match network output.
		error = label - prediction
		if self.cost_fn == 'mse':
			if not derivative:
				return ( np.square( error ) ).sum() / error.size
			else:
				return ( 2 * error ).sum() / error.size

	def evaluate(self,X):
		for layer in self.structure:
			X = layer._forwards(X)
		return X

	def save_model(self,name: str):
		assert name.split('.')[-1] == 'pkl'
		with open(name, 'wb') as file:  
			pickle.dump(self, file)

	@staticmethod
	def load_model(name):
		assert name.split('.')[-1] == 'pkl'
		with open(name, 'rb') as file:  
			model = pickle.load(file)
		return model
		

	class Conv_Layer:
		def __init__(self,filt_shape: tuple,num_filters: int=5,stride: int=1,padding: int=0,pad_type: str=None,random_seed=42):
			""" 
			- filt_shape (tuple): A tuple object describing the 2D shape of the filter to be convolved over the input.
			- num_filters (int): Number of filters to be used.
			- stride (int): Size of steps to take when shifting the filter. (Currently stride_x = stride_y).
			- padding (int): Width of zero-padding to apply on each side of the array. Only applied if pad_type is None.
			- pad_type (str): Options: same (output shape is same as input shape), valid (equal to padding=0), include (padding added evenly on all sides of the array to allow the filter to shift over the input an integer number of times - avoid excluding input data).
			- random_seed (int): The seed provided to numpy before initiating the filters and biases. random_seed=None will result in no seed being provided meaning numpy will generate it dynamically each time.
			"""
			self.model = None

			self.LAYER_TYPE = 'CONV'
			self.FILT_SHAPE = filt_shape	# 2D tuple describing num rows and cols
			self.NUM_FILTERS = num_filters
			self.STRIDE = stride
			self.PADDING = padding
			if pad_type:
				self.PAD_TYPE = pad_type.lower()
			else:
				self.PAD_TYPE = None
			self.RANDOM_SEED = random_seed

			self.next_layer = None
			self.prev_layer = None

			self.output = None

		def prepare_layer(self):
			if self.prev_layer == None:	# This means this is the first layer in the structure, so 'input' is the only thing before.
				INPUT_SHAPE = self.model.INPUT_SHAPE		# (channels, rows, cols)
			else:
				INPUT_SHAPE = self.prev_layer.output.shape		# (channels, rows, cols)

			assert len(INPUT_SHAPE) in (2,3), 'Invalid INPUT_SHAPE'

			# Convert 2D input to 3D.
			if len(INPUT_SHAPE) == 2:
				INPUT_SHAPE = tuple([1]) + INPUT_SHAPE	

			NUM_INPUT_ROWS = INPUT_SHAPE[-2]
			NUM_INPUT_COLS = INPUT_SHAPE[-1]

			# Initiate filters
			filts = []
			for _ in range(self.NUM_FILTERS):
				np.random.seed(self.RANDOM_SEED)
				filts.append( np.random.normal(size=(INPUT_SHAPE[0],self.FILT_SHAPE[0], self.FILT_SHAPE[1]) ) )
			self.filters = np.array(filts)
			np.random.seed(self.RANDOM_SEED)
			self.bias = np.random.normal(size=(self.NUM_FILTERS,1))

			# Need to account for padding.
			if self.PAD_TYPE != None:
				if self.PAD_TYPE == 'same':
					nopad_out_cols = math.ceil(float(NUM_INPUT_COLS) / float(self.STRIDE))
					pad_cols_needed = max((nopad_out_cols - 1) * self.STRIDE + self.FILT_SHAPE[1] - NUM_INPUT_COLS, 0)
					nopad_out_rows = math.ceil(float(NUM_INPUT_ROWS) / float(self.STRIDE))
					pad_rows_needed = max((nopad_out_rows - 1) * self.STRIDE + self.FILT_SHAPE[0] - NUM_INPUT_ROWS, 0)
				elif self.PAD_TYPE == 'valid':
					# TensoFlow definition of this is "no padding". The input is just processed as-is.
					pad_rows_needed = pad_cols_needed = 0
				elif self.PAD_TYPE == 'include':
					# Here we will implement the padding method to avoid input data being excluded/ missed by the convolution.
					# - This happens when, (I_dim - F_dim) % stride != 0
					if (self.NUM_INPUT_ROWS - self.FILT_SHAPE[0]) % self.STRIDE != 0:
						pad_rows_needed = self.FILT_SHAPE[0] - ((self.NUM_INPUT_ROWS - self.FILT_SHAPE[0]) % self.STRIDE)
					else:
						pad_rows_needed = 0
					if (self.NUM_INPUT_COLS - self.FILT_SHAPE[1]) % self.STRIDE != 0:
						pad_cols_needed = self.FILT_SHAPE[1] - ((self.NUM_INPUT_COLS - self.FILT_SHAPE[1]) % self.STRIDE)
					else:
						pad_cols_needed = 0

				self.COL_LEFT_PAD = pad_cols_needed // 2	# // Floor division
				self.COL_RIGHT_PAD = math.ceil(pad_cols_needed / 2)
				self.ROW_UP_PAD = pad_rows_needed // 2	# // Floor division
				self.ROW_DOWN_PAD = math.ceil(pad_rows_needed / 2)
			else:
				self.COL_LEFT_PAD = self.COL_RIGHT_PAD = self.ROW_UP_PAD = self.ROW_DOWN_PAD = self.PADDING

			col_out = int((NUM_INPUT_COLS + (self.COL_LEFT_PAD + self.COL_RIGHT_PAD) - self.FILT_SHAPE[1]) / self.STRIDE) + 1
			row_out = int((NUM_INPUT_ROWS + (self.ROW_DOWN_PAD + self.ROW_UP_PAD) - self.FILT_SHAPE[0]) / self.STRIDE) + 1

			self.output = np.zeros(shape=(self.NUM_FILTERS,row_out,col_out))	# Output initiated.
			if self.PAD_TYPE == 'same':
				assert self.output.shape[-2:] == self.INPUT_SHAPE[-2:]	# Channels may differ.

		def define_details(self):
			return {
				'LAYER_TYPE':self.LAYER_TYPE,
				'NUM_FILTERS':self.NUM_FILTERS,
				'STRIDE':self.STRIDE
			}

		def _forwards(self,_input):
			if _input.ndim == 3:
				self.input = _input
			elif _input.ndim == 2:
				self.input = np.array( [ _input ] )	# NOTE: 'fakes' number of channels to be 1.

			# Apply the padding to the input.
			self.padded_input = np.pad(self.input,[(0,0),(self.ROW_UP_PAD,self.ROW_DOWN_PAD),(self.COL_LEFT_PAD,self.COL_RIGHT_PAD)],'constant')

			proc_rows, proc_cols = self.padded_input.shape[-2:]

			for filt_index in range(self.NUM_FILTERS):
				filt = self.filters[filt_index]
				
				filt_channels, filt_rows, filt_cols = filt.shape

				for channel_index in range(filt_channels):
					self.output[filt_index] += CNN.Conv_Layer.convolve( self.padded_input[channel_index], filt[channel_index], self.STRIDE )
				
				self.output[filt_index] += self.bias[filt_index]

			return self.output	# NOTE: Output is 3D array of shape: ( NUM_FILTS, NUM_ROWS, NUM_COLS )

		def _backwards(self,cost_gradient):	
			assert cost_gradient.shape == self.output.shape, f'cost_gradient shape [{cost_gradient.shape}] does not match layer output shape [{self.output.shape}].'

			_, c_rows, c_cols = cost_gradient.shape
			dilation_idx_row = np.arange(c_rows-1) + 1	# Intiatial indices for insertion of zeros
			dilation_idx_col = np.arange(c_cols-1) + 1	# Intiatial indices for insertion of zeros
			cost_gradient_dilated = cost_gradient
			for n in range(1,self.STRIDE):
				cost_gradient_dilated = np.insert(
					np.insert( cost_gradient_dilated, dilation_idx_row * n, 0, axis=1 ),
					dilation_idx_col * n, 0, axis=2)	# the n multiplier is to increment the indices in the non-uniform manner required.
			print(f'cost_gradient shape: {cost_gradient.shape} | cost_gradient_dilated shape: {cost_gradient_dilated.shape}')

			dCdF = []	# initiate as list then convert to np.array
			dCdX_pad_excl = []
			# Find cost gradient wrt previous output.
			rotated_filters = np.rot90( self.filters, k=2, axes=(1,2) )	# rotate 2x90 degs, rotating in direction of rows to columns.
			for r_filt_ind in range(rotated_filters.shape[0]):
				filt_1_container = []
				filt_2_container = []
				for channel_index in range(self.padded_input.shape[0]):
					# dCdF
					filt_1_container.append( CNN.Conv_Layer.convolve( self.padded_input[channel_index], cost_gradient_dilated[r_filt_ind], stride=1 ) )
					# dCdX
					filt_2_container.append( CNN.Conv_Layer.convolve( cost_gradient_dilated[r_filt_ind], rotated_filters[r_filt_ind][channel_index], stride=1, full_convolve=True ) )
				dCdF.append(filt_1_container)
				dCdX_pad_excl.append(filt_2_container)
			dCdF = np.array( dCdF )
			dCdX_pad_excl = np.array( dCdX_pad_excl ).sum(axis=0)	# NOTE: This is the cost gradient w.r.t. the padded input and potentially excluding pixels.
			
			# Account for filter not shifting over input an integer number of times with given stride.
			pxls_excl_x = (self.padded_input.shape[2] - self.FILT_SHAPE[1]) % self.STRIDE	# pixels excluded in x direction (cols)
			pxls_excl_y = (self.padded_input.shape[1] - self.FILT_SHAPE[0]) % self.STRIDE	# pixels excluded in y direction (rows)
			dCdF = dCdF[:,:, : dCdF.shape[2] - pxls_excl_y, : dCdF.shape[3] - pxls_excl_x]	# Remove the values from right and bottom of array (this is where the excluded pixels will be).
			dCdX_pad = np.zeros(shape=self.padded_input.shape)
			dCdX_pad[:,: dCdX_pad.shape[1] - pxls_excl_y, : dCdX_pad.shape[2] - pxls_excl_x] = dCdX_pad_excl	# pixels excluded in forwards pass will now appear with cost_gradient = 0.
			assert dCdF.shape == self.filters.shape, f'dCdF shape [{dCdF.shape}] does not match filters shape [{self.filters.shape}].'

			# Remove padding that was added to the input array.
			dCdX = dCdX_pad[ : , self.ROW_UP_PAD : dCdX_pad.shape[1] - self.ROW_DOWN_PAD , self.COL_LEFT_PAD : dCdX_pad.shape[2] - self.COL_RIGHT_PAD ]
			assert dCdX.shape == self.input.shape, f'dCdX shape [{dCdX.shape}] does not match layer input shape [{self.input.shape}].'

			self.filters = self.filters + ( self.model.LEARNING_RATE * dCdF	) # ADJUST THE FILTERS
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

	
	class Pool_Layer:
		def __init__(self,filt_shape: tuple,stride: int,pool_type: str='max',padding: int=0,pad_type: str=None):
			'''
			- filt_shape (tuple): A tuple object describing the 2D shape of the filter to use for pooling.
			- stride (int): Size of steps to take when shifting the filter. (Currently stride_x = stride_y).
			- pool_type (str): Pooling method to be applied. Options = max, min, mean.
			- padding (int): Width of zero-padding to apply on each side of the array. Only applied if pad_type is None.
			- pad_type (str): Options: same (output shape is same as input shape), valid (equal to padding=0), include (padding added evenly on all sides of the array to allow the filter to shift over the input an integer number of times - avoid excluding input data).
			'''
			self.model = None

			self.LAYER_TYPE = 'POOL'
			self.FILT_SHAPE = filt_shape	# 2D array (rows,cols)
			self.STRIDE = stride
			self.POOL_TYPE = pool_type.lower()
			assert self.POOL_TYPE in ('max','mean','min')
			self.PADDING = padding
			self.PAD_TYPE = None if pad_type is None else pad_type.lower()

			self.next_layer = None
			self.prev_layer = None

		def prepare_layer(self):
			""" This needs to be done after the input has been identified - currently happens when train() is called. """
			if self.prev_layer == None:	# This means this is the first layer in the structure, so 'input' is the only thing before.
				INPUT_SHAPE = self.model.INPUT_SHAPE		# (channels, rows, cols)
			else:
				INPUT_SHAPE = self.prev_layer.output.shape		# (channels, rows, cols)

			assert len(INPUT_SHAPE) in (2,3), 'Invalid INPUT_SHAPE'

			# Convert 2D input to 3D.
			if len(INPUT_SHAPE) == 2:
				INPUT_SHAPE = tuple([1]) + INPUT_SHAPE

			NUM_INPUT_ROWS = INPUT_SHAPE[-2]
			NUM_INPUT_COLS = INPUT_SHAPE[-1]

			# Need to account for padding.
			if self.PAD_TYPE != None:
				if self.PAD_TYPE == 'same':
					nopad_out_cols = math.ceil(float(NUM_INPUT_COLS) / float(self.STRIDE))
					pad_cols_needed = max((nopad_out_cols - 1) * self.STRIDE + self.FILT_SHAPE[1] - NUM_INPUT_COLS, 0)
					nopad_out_rows = math.ceil(float(NUM_INPUT_ROWS) / float(self.STRIDE))
					pad_rows_needed = max((nopad_out_rows - 1) * self.STRIDE + self.FILT_SHAPE[0] - NUM_INPUT_ROWS, 0)
				elif self.PAD_TYPE == 'valid':
					# TensoFlow definition of this is "no padding". The input is just processed as-is.
					pad_rows_needed = pad_cols_needed = 0
				elif self.PAD_TYPE == 'include':
					# Here we will implement the padding method to avoid input data being excluded/ missed by the convolution.
					# - This happens when, (I_dim - F_dim) % stride != 0
					if (self.NUM_INPUT_ROWS - self.FILT_SHAPE[0]) % self.STRIDE != 0:
						pad_rows_needed = self.FILT_SHAPE[0] - ((self.NUM_INPUT_ROWS - self.FILT_SHAPE[0]) % self.STRIDE)
					else:
						pad_rows_needed = 0
					if (self.NUM_INPUT_COLS - self.FILT_SHAPE[1]) % self.STRIDE != 0:
						pad_cols_needed = self.FILT_SHAPE[1] - ((self.NUM_INPUT_COLS - self.FILT_SHAPE[1]) % self.STRIDE)
					else:
						pad_cols_needed = 0

				self.COL_LEFT_PAD = pad_cols_needed // 2	# // Floor division
				self.COL_RIGHT_PAD = math.ceil(pad_cols_needed / 2)
				self.ROW_UP_PAD = pad_rows_needed // 2	# // Floor division
				self.ROW_DOWN_PAD = math.ceil(pad_rows_needed / 2)
			else:
				self.COL_LEFT_PAD = self.COL_RIGHT_PAD = self.ROW_UP_PAD = self.ROW_DOWN_PAD = self.PADDING

			col_out = int((NUM_INPUT_COLS + (self.COL_LEFT_PAD + self.COL_RIGHT_PAD) - self.FILT_SHAPE[1]) / self.STRIDE) + 1
			row_out = int((NUM_INPUT_ROWS + (self.ROW_DOWN_PAD + self.ROW_UP_PAD) - self.FILT_SHAPE[0]) / self.STRIDE) + 1

			self.output = np.zeros(shape=(INPUT_SHAPE[0],row_out,col_out))	# Output initiated.
			if self.PAD_TYPE == 'same':
				assert self.output.shape[-2:] == self.INPUT_SHAPE[-2:]	# Channels may differ.

		def define_details(self):
			return {
				'LAYER_TYPE':self.LAYER_TYPE,
				'STRIDE':self.STRIDE,
				'POOL_TYPE':self.POOL_TYPE
			}

		def _forwards(self,_input):
			if _input.ndim == 3:
				self.input = _input
			elif _input.ndim == 2:
				self.input = np.array( [ _input ] )	# NOTE: 'fakes' number of channels to be 1.

			assert self.input.ndim == 3

			# Apply the padding to the input.
			self.padded_input = np.pad(self.input,[(0,0),(self.ROW_UP_PAD,self.ROW_DOWN_PAD),(self.COL_LEFT_PAD,self.COL_RIGHT_PAD)],'constant')

			channels, proc_rows, proc_cols = self.padded_input.shape

			# Shift 'Filter Window' over the image and perform the downsampling
			curr_y = out_y = 0
			while curr_y <= proc_rows - self.FILT_SHAPE[0]:
				curr_x = out_x = 0
				while curr_x <= proc_cols - self.FILT_SHAPE[1]:
					for channel_index in range(channels):
						if self.POOL_TYPE == 'max':
							sub_arr = self.padded_input[ channel_index, curr_y : curr_y + self.FILT_SHAPE[0], curr_x : curr_x+ self.FILT_SHAPE[1] ]
							self.output[channel_index, out_y, out_x] = np.max( sub_arr )
						elif self.POOL_TYPE == 'min':
							sub_arr = self.padded_input[ channel_index, curr_y : curr_y + self.FILT_SHAPE[0], curr_x : curr_x+ self.FILT_SHAPE[1] ]
							self.output[channel_index, out_y, out_x] = np.min( sub_arr )
						elif self.POOL_TYPE == 'mean':
							sub_arr = self.padded_input[ channel_index, curr_y : curr_y + self.FILT_SHAPE[0], curr_x : curr_x + self.FILT_SHAPE[1] ]
							self.output[channel_index, out_y, out_x] = np.mean( sub_arr )

					curr_x += self.STRIDE
					out_x += 1
				curr_y += self.STRIDE
				out_y += 1

			return self.output

		def _backwards(self,cost_gradient):
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
			# Initiate to input shape.
			prev_cost_gradient = np.zeros_like(self.padded_input)

			assert cost_gradient.shape[0] == prev_cost_gradient.shape[0]

			channels, rows, cols = prev_cost_gradient.shape

			# Step over the array similarly to the forwards pass and compute the expanded cost gradients.
			curr_y = cost_y = 0
			while curr_y <= rows - self.FILT_SHAPE[0]:
				curr_x = cost_x = 0
				while curr_x <= cols - self.FILT_SHAPE[1]:
					for channel_index in range(channels):
						if self.POOL_TYPE == 'max':
							# Set value of node that corresponds with the max value node of the input to the cost gradient value at (cost_y,cost_x)
							sub_arr = self.padded_input[ channel_index, curr_y : curr_y + self.FILT_SHAPE[0], curr_x : curr_x + self.FILT_SHAPE[1] ]
							max_node_y, max_node_x = np.array( np.unravel_index( np.argmax( sub_arr ), sub_arr.shape ) ) + np.array([curr_y, curr_x])	# addition of curr_y & curr_x is to get position in padded_input array (not just local sub_arr).

							cost_val = cost_gradient[channel_index,cost_y,cost_x]

							prev_cost_gradient[channel_index, max_node_y, max_node_x] += cost_val
						elif self.POOL_TYPE == 'min':
							# Set value of node that corresponds with the min value node of the input to the cost gradient value at (cost_y,cost_x)
							sub_arr = self.padded_input[ channel_index, curr_y : curr_y + self.FILT_SHAPE[0], curr_x : curr_x + self.FILT_SHAPE[1] ]
							min_node_y, min_node_x = np.array( np.unravel_index( np.argmin( sub_arr ), sub_arr.shape ) ) + np.array([curr_y, curr_x])	# addition of curr_y & curr_x is to get position in padded_input array (not just local sub_arr).

							cost_val = cost_gradient[channel_index,cost_y,cost_x]

							prev_cost_gradient[channel_index, min_node_y, min_node_x] += cost_val
						elif self.POOL_TYPE == 'mean':
							sub_arr = self.padded_input[ channel_index, curr_y : curr_y + self.FILT_SHAPE[0], curr_x : curr_x + self.FILT_SHAPE[1] ]

							cost_val = cost_gradient[channel_index,cost_y,cost_x]
							
							sub_arr_props = sub_arr / sub_arr.sum()

							prev_cost_gradient[ channel_index, curr_y : curr_y + self.FILT_SHAPE[0], curr_x : curr_x + self.FILT_SHAPE[1] ] += sub_arr_props * cost_val

					curr_x += self.STRIDE
					cost_x += 1
				curr_y += self.STRIDE
				cost_y += 1

			return prev_cost_gradient


	class Flatten_Layer:
		""" A psuedo layer that simply adjusts the data dimension as it passes between 2D/3D Conv or Pool layers to the 1D FC layers. """

		def __init__(self):
			self.LAYER_TYPE = 'FLATTEN'

			self.next_layer = None
			self.prev_layer = None

		def prepare_layer(self):
			self.output = np.zeros(shape=(self.prev_layer.output.size,1))

		def define_details(self):
			return {
				'LAYER_TYPE':self.LAYER_TYPE
			}

		def _forwards(self,_input):
			return _input.reshape((_input.size,1))	# NOTE: Vertical array

		def _backwards(self,cost_gradient):
			return cost_gradient.reshape(self.prev_layer.output.shape)


	class FC_Layer:
		"""
		The Fully Connected Layer is defined as being the layer of nodes and the weights of the connections that link those nodes to the previous layer.
		"""
		def __init__(self, num_nodes, activation: str=None,random_seed=42):
			"""
			- n: Number of nodes in layer.
			- activation: The name of the activation function to be used. The activation is handled by a CNN.Activation_Layer object that is transparent to the user here. Defaults to None - a transparent Activation layer will still be added however, the data passing through will be untouched.
			"""
			self.model = None

			self.LAYER_TYPE = 'FC'
			self.NUM_NODES = num_nodes
			self.ACTIVATION = None if activation is None else activation.lower()
			self.RANDOM_SEED = random_seed

			self.next_layer = None
			self.prev_layer = None

			self.output = np.zeros(shape=(num_nodes,1))	# NOTE: This is a vertical array.

		def prepare_layer(self):
			""" Initiate weights and biases randomly"""
			w_cols = self.prev_layer.output.size
			w_rows = self.NUM_NODES	# NOTE: "Each row corresponds to all connections of previous layer to a single node in current layer." - based on vertical node array.
			np.random.seed(self.RANDOM_SEED)
			self.weights = np.random.normal(size=(w_rows,w_cols))	# NOTE: this is the correct orientation for vertical node array.
			
			np.random.seed(self.RANDOM_SEED)
			self.bias = np.random.normal(size=(self.NUM_NODES,1))	# NOTE: MUST be same shape as output array.

		def define_details(self):
			return {
				'LAYER_TYPE':self.LAYER_TYPE,
				'NUM_NODES':self.NUM_NODES,
				'ACTIVATION':self.ACTIVATION
			}

		def _forwards(self,_input):
			self.input = _input

			self.output = np.dot( self.weights, self.input ) + self.bias

			return self.output

		def _backwards(self, cost_gradient):
			"""
			Take cost gradient dC/dZ (how the output of this layer affects the cost) and backpropogate

			Z = W . I + B

			cost gradient shape === Z shape
			"""
			assert cost_gradient.shape == self.output.shape

			Z = self.output	# Weighted sum is calculated on forwards pass.
			dZ_dW = self.input	# Partial diff of weighted sum (Z) w.r.t. weights
			dZ_dB = 1
			dZ_dI = np.transpose( self.weights )	# Partial diff of weighted sum w.r.t. input to layer.
			
			dC_dW = np.multiply( cost_gradient , np.transpose( dZ_dW ) )	# Element-wise multiplication. The local gradient needs transposing for the multiplication.

			self.weights = self.weights + ( self.model.LEARNING_RATE * dC_dW )

			dC_dB = np.multiply( cost_gradient, dZ_dB )	# Element-wise multiplication

			self.bias = self.bias + ( self.model.LEARNING_RATE * dC_dB )

			return np.matmul( dZ_dI , cost_gradient )	# Matrix multiplication


	class Activation:
		def __init__(self,function: str=None):
			self.model = None

			self.LAYER_TYPE = 'ACTIVATION'

			self.next_layer = None
			self.prev_layer = None

			self.FUNCTION = None if function is None else function.lower()

		def prepare_layer(self):
			self.output = np.zeros(shape=self.prev_layer.output.shape )

		def define_details(self):
			return {
				'LAYER_TYPE':self.LAYER_TYPE,
				'FUNCTION':self.FUNCTION
			}

		def _forwards(self,_input):
			self.input = _input
			
			if self.FUNCTION is None:
				self.output = _input
			elif self.FUNCTION == 'relu':
				# The ReLu function is highly computationally efficient but is not able to process inputs that approach zero or negative.
				_input[_input<0] = 0
				self.output = _input
			elif self.FUNCTION == 'softmax':
				# Softmax is a special activation function use for output neurons. It normalizes outputs for each class between 0 and 1, and returns the probability that the input belongs to a specific class.
				exp = np.exp(_input)
				self.output = exp / np.sum(exp)
			elif self.FUNCTION == 'sigmoid':
				# The sigmoid function has a smooth gradient and outputs values between zero and one. For very high or low values of the input parameters, the network can be very slow to reach a prediction, called the vanishing gradient problem.
				self.output = 1 / (1 + np.exp(-_input))
			elif self.FUNCTION == 'step': # TODO: Define "step function" activation
				pass
			elif self.FUNCTION == 'tanh':
				# The TanH function is zero-centered making it easier to model inputs that are strongly negative strongly positive or neutral.
				self.output = ( np.exp(_input) - np.exp(-_input) ) / ( np.exp(_input) + np.exp(-_input) )
			elif self.FUNCTION == 'swish': # TODO: Define "Swish function" activation
				# Swish is a new activation function discovered by Google researchers. It performs better than ReLu with a similar level of computational efficiency.
				pass
			elif self.FUNCTION == 'leaky relu':
				# The Leaky ReLu function has a small positive slope in its negative area, enabling it to process zero or negative values.
				self.alpha = 0.01
				_input[_input <= 0] = self.alpha * _input[_input <= 0]
				self.output = _input
			elif self.FUNCTION == 'parametric relu': # TODO: Define "Parametric ReLu"
				#  The Parametric ReLu function allows the negative slope to be learned, performing backpropagation to learn the most effective slope for zero and negative input values.
				pass
			return self.output

		def _backwards(self,cost_gradient):
			# NOTE: Differation of Activation w.r.t. z
			dA_dZ = None
			if self.FUNCTION is None: # a = z
				dA_dZ = np.ones( self.input.shape )
			elif self.FUNCTION == 'relu':
				dA_dZ = self.input
				dA_dZ[dA_dZ <= 0] = 0
				dA_dZ[dA_dZ> 0] = 1
			elif self.FUNCTION == 'softmax': # TODO
				pass
			elif self.FUNCTION == 'sigmoid':
				dA_dZ =  self.output * (1 - self.output)	# Element-wise multiplication.
			elif self.FUNCTION == 'step': # TODO: Define "step function" derivative
				pass
			elif self.FUNCTION == 'tanh':
				dA_dZ =  1 - np.square( self.output )
			elif self.FUNCTION == 'swish': # TODO: Define "Swish function" derivative
				pass
			elif self.FUNCTION == 'leaky relu':
				dA_dZ = self.input
				dA_dZ[dA_dZ > 0] = 1
				dA_dZ[dA_dZ <= 0] = self.alpha
			elif self.FUNCTION == 'parametric relu': # TODO: Define "Parametric ReLu" derivative
				pass
			
			assert dA_dZ is not None, f'WARNING:: No derivative defined for chosen activation function "{self.FUNCTION}"'
			
			return np.multiply( dA_dZ , cost_gradient )	# Element-wise multiplication.

