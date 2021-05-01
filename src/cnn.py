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
import sys

# CLASS
class CNN():
	"""
	This is the top level class. It contains sub-classes for each of the layers that are to be included in the model.
	"""

	def __init__(self,input_shape: tuple,optimiser_method='gd'):
		'''
		- optimiser_method (str): Options: ('gd','momentum','rmsprop','adam'). Default is 'gd'.
		'''
		assert len(input_shape) == 3, 'input_shape must be of length 3: (num_channels, num_rows, num_columns)'
		assert optimiser_method.lower() in CNN.SUPPORTED_OPTIMISERS, f'You must provide an optimiser that is supported. The options are: {CNN.SUPPORTED_OPTIMISERS}'

		self.is_prepared = False

		self.INPUT_SHAPE = input_shape	# tuple to contain input shape
		self.OPTIMISER_METHOD = optimiser_method.lower()

		self.structure = []	# defines order of model (list of layer objects) - EXCLUDES INPUT DATA
		self.layer_counts = {'total':0,'CONV':0,'POOL':0,'FLATTEN':0,'FC':0,'ACTIVATION':0}	# dict for counting number of each layer type

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
		self.layer_counts[layer.LAYER_TYPE] += 1
		self.layer_counts['total'] += 1

		if layer.LAYER_TYPE == 'FC':
			# Create the Activation Layer (transparent to user).
			self.add_layer(
				CNN.Activation(function=layer.ACTIVATION)
			)

	def remove_layer(self,index):
		self.structure.pop(index)
		if self.is_prepared:
			print('-- INFO:: Re-compiling model...')
			self.prepare_model()
			
	def get_model_details(self):
		details = []
		for layer in self.structure:
			details.append(layer.define_details())

		return details
		
	def prepare_model(self):
		""" Called once final layer is added, each layer can now iniate its weights and biases. """
		print('Preparing model...')
		if self.layer_counts['total'] > 1:
			for index in range(self.layer_counts['total']):
				curr_layer = self.structure[index]
				if index != len(self.structure) - 1:
					next_layer = self.structure[index + 1]
				else:
					next_layer = None

				curr_layer.next_layer = next_layer
				if next_layer is not None:
					next_layer.prev_layer = curr_layer

				curr_layer.MODEL_STRUCTURE_INDEX = index

				print(f'Preparing Layer:: Type = {curr_layer.LAYER_TYPE} | Structure index = {curr_layer.MODEL_STRUCTURE_INDEX}')
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

	def train(self,Xs,ys,epochs,max_batch_size=32,shuffle=False,random_seed=42,learning_rate=0.001,cost_fn='mse',beta1=0.9,beta2=0.999):
		'''
		Should take array of inputs and array of labels of the same length.

		[For each epoch] For each input, propogate forwards and backwards.

		ARGS:
		- Xs (np.ndarray or list): (N,ch,rows,cols). Where N is number of examples, ch is number of channels.
		- ys (np.ndarray or list): (N,num_categories). e.g. ex1 label = [0,0,0,1,0] for 5 categories (one-hot encoded).
		- epochs (int): Number of iterations over the data set.
		- max_batch_size (int): Maximum number of examples in each batch. (Tensorflow defaults to 32. Also, batch_size > N will be truncated.)
		- shuffle (bool): Determines whether the data set is shuffled before training.
		- random_seed (int): The seed provided to numpy before performing the shuffling. random_seed=None will result in no seed being provided meaning numpy will generate it dynamically each time.
		- beta1 (float): param used for Adam optimisation
		- beta2 (float): param used for Adam optimisation
		'''
		ys = ys.reshape(-1,1) if ys.ndim == 1 else ys
		# --------- ASSERTIONS -----------
		# Check shapes and orientation are as expected
		assert Xs.shape[0] == ys.shape[0], 'Dimension of input data and labels does not match.'
		assert ys.shape[-1] == self.structure[-1].output.shape[0], 'Invalid shape for labels. Should be (N,num_categories)'	# NOTE: This assumes last layer in model have output that is a vertical array.
		assert type(epochs) == int
		assert int(max_batch_size) == max_batch_size and max_batch_size is not None, 'An integer value must be supplied for argument "max_batch_size"'
		assert cost_fn.lower() in CNN.SUPPORTED_COST_FUNCTIONS, f'Chosen cost function not supported, please choose: {CNN.SUPPORTED_COST_FUNCTIONS}'
		
		# --------- ASSIGNMENTS ----------
		self.N = Xs.shape[0]	# Total number of examples in Xs
		Xs, ys = np.array(Xs), np.array(ys)	# Convert data to numpy arrays in case not already.
		self.Xs, self.ys = CNN.shuffle(Xs,ys,random_seed) if shuffle else Xs, ys
		self.EPOCHS = epochs
		if self.OPTIMISER_METHOD == 'sgd':
			self.MAX_BATCH_SIZE = 1
		else:
			self.MAX_BATCH_SIZE = self.N if max_batch_size > self.N else max_batch_size
		self.BATCH_COUNT = math.ceil( self.N / self.MAX_BATCH_SIZE )
		self.COST_FN = cost_fn.lower()
		self.LEARNING_RATE = learning_rate
		self.BETA1 = beta1
		self.BETA2 = beta2
		self.feed_forwards_cycle_index = -1
		self.iteration_index = -1	# incremented at start of each backprop so needs to be initiated to -1
		self.iteration_cost = 0
		self.iteration_cost_gradient = 0

		if not self.is_prepared:
			self.prepare_model()

		self.TOTAL_ITERATIONS = self.BATCH_COUNT * self.EPOCHS

		self._initiate_tracking_metrics()

		# ---------- TRAIN -------------
		train_start = dt.now()
		for epoch_ind in range(self.EPOCHS):
			self.epoch_ind = epoch_ind
			
			self._iterate_forwards()

		return dt.now(), dt.now() - train_start	# returns training finish time and duration.

	SUPPORTED_OPTIMISERS = ('gd','momentum','rmsprop','adam')

	def _iterate_forwards(self):
		for batch_ind in range(self.BATCH_COUNT):
			ind_lower = batch_ind * self.MAX_BATCH_SIZE	# Lower bound of index range
			ind_upper = batch_ind * self.MAX_BATCH_SIZE + self.MAX_BATCH_SIZE	# Upper bound of index range
			if ind_upper > self.N and self.N > 1:
				ind_upper = self.N

			# print('Lower index:',ind_lower,'Upper index:',ind_upper)

			batch_Xs = self.Xs[ ind_lower : ind_upper ]
			batch_ys = self.ys[ ind_lower : ind_upper ]

			batch_size = len(batch_Xs)

			for ex_ind , X in enumerate(batch_Xs):	# For each example (observation)
				print(X.shape)
				prediction = self.predict(X,training=True)

				self.iteration_cost += self.cost(prediction, batch_ys[ex_ind],batch_size=batch_size)
				self.iteration_cost_gradient += self.cost(prediction, batch_ys[ex_ind],batch_size=batch_size,derivative=True)

			print(f'-- Epoch: {self.epoch_ind+1}/{self.EPOCHS } | Batch: {batch_ind+1}/{self.BATCH_COUNT} | Cost: {self.iteration_cost}')

			self._iterate_backwards(self.iteration_cost_gradient)

	def _iterate_backwards(self,cost_gradient):
		self.iteration_index += 1
		self.history['cost'][self.iteration_index] = self.iteration_cost
		# Backpropagate the cost_gradient
		for layer in self.structure[::-1]:
			cost_gradient = layer._backwards(cost_gradient)

		self.iteration_cost = 0
		self.iteration_cost_gradient = 0

	def predict(self,X,training=False):
		if training: self.feed_forwards_cycle_index += 1
		for layer in self.structure:
			X = layer._forwards(X)
			# print('Layer index:',layer.MODEL_STRUCTURE_INDEX)
			# print('Output:',X)
		return X

	@staticmethod
	def shuffle(X,y,random_seed=None):
		if random_seed is not None:
			np.random.seed(random_seed)
		permutation = np.random.permutation( self.N )

		self.Xs = Xs[permutation]
		self.ys = ys[permutation]

	def _initiate_tracking_metrics(self):
		# Initiate model history tracker
		initial_arr = np.zeros(self.TOTAL_ITERATIONS)
		initial_arr[:] = np.NaN
		self.history = {'cost':initial_arr}	# dict object will allow us to store history of various parameters.
		
		# Initiate layer history trackers
		for layer in self.structure:
			layer._initiate_history()

	SUPPORTED_COST_FUNCTIONS = ('mse','cross_entropy')

	def cost(self,prediction,label,batch_size,derivative=False):
		'''
		Cost function to provide measure of model 'correctness'. returns vector cost value.
		'''
		print(label)
		label = label.reshape((max(label.shape),1))	# reshape label to vertical array to match network output.
		error = label - prediction	# Vector
		if self.COST_FN == 'mse':
			if not derivative:
				return np.sum( np.square( error ) ) / (batch_size * prediction.size)	# Vector
			else:
				return -( 2 * error ) / batch_size	# Vector
		elif self.COST_FN == 'cross_entropy':
			if not derivative:
				print('logprobs:',np.log(prediction))
				cost = -np.sum(label * np.log(prediction)) / batch_size
				print('Cost:',cost)
				return cost
			else:
				return -(label/prediction) / batch_size
				# return prediction - 

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

	@staticmethod
	def array_init(shape,method=None):
		''' Random initialisation of weights array.
		Xavier or Kaiming: (https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79) '''
		assert len(shape) >= 2
		fan_in = shape[-1]
		fan_out = shape[-2]
		if method is None:
			array = np.random.rand(fan_out,fan_in) * 0.01
		elif method == 'kaiming_normal':
			# AKA "he_normal" after Kaiming He.
			array = np.random.normal(size=shape) * np.sqrt(2./fan_in)
		elif method == 'kaiming_uniform':
			array = np.random.uniform(size=shape) * np.sqrt(6./fan_in)
		elif method == 'xavier_uniform':
			array = np.random.uniform(size=shape) * np.sqrt(6./(fan_in+fan_out))
		elif method == 'xavier_normal':
			# https://arxiv.org/pdf/2004.09506.pdf
			target_std = np.sqrt(2./np.sum(shape))
			array = np.random.normal(size=shape,scale=target_std)
		elif method == 'abs_norm':
			# Custom alternative
			arr = np.random.normal(size=shape)
			array = arr / np.abs(arr).max()
		elif method == 'uniform':
			array = np.random.uniform(size=shape) * (1./np.sqrt(fan_in))

		print(f'Array init method: {method}, max: {array.max()}, min: {array.min()}, std: {array.std()}' )
		print('Array:',array)
		return array


	class CNN_Layer:
		'''
		PARENT LAYER CLASS FOR ALL LAYER TYPES
		'''
		def __init__(self):
			self.model = None

			self.next_layer = None
			self.prev_layer = None

			self.output = None

		def _initiate_history(self):
			out_init_arr = np.zeros(self.model.EPOCHS * self.model.N)
			out_init_arr[:] = np.NaN
			cg_init_arr = np.zeros(self.model.TOTAL_ITERATIONS)
			cg_init_arr[:] = np.NaN
			self.history = {
				'output':
					{'mean':out_init_arr,'std':out_init_arr,'max':out_init_arr,'min':out_init_arr,'sum':out_init_arr,'median':out_init_arr},
				'cost_gradient':
					{'mean':cg_init_arr,'std':cg_init_arr,'max':cg_init_arr,'min':cg_init_arr,'sum':cg_init_arr,'median':cg_init_arr}
			}

		def _track_metrics(self,output=None,cost_gradient=None):
			if output is not None:
				self.history['output']['mean'][self.model.feed_forwards_cycle_index] = np.mean(output)
				self.history['output']['std'][self.model.feed_forwards_cycle_index] = np.std(output)
				self.history['output']['max'][self.model.feed_forwards_cycle_index] = np.max(output)
				self.history['output']['min'][self.model.feed_forwards_cycle_index] = np.min(output)
				self.history['output']['sum'][self.model.feed_forwards_cycle_index] = np.sum(output)
				self.history['output']['median'][self.model.feed_forwards_cycle_index] = np.median(output)
			if cost_gradient is not None:
				if not np.isnan(self.history['cost_gradient']['mean'][self.model.iteration_index]):
					print(f"Warning: value already set. Overwriting {self.history['cost_gradient']['mean'][self.model.iteration_index]} with {np.mean(cost_gradient)}")
					sys.exit()
				self.history['cost_gradient']['mean'][self.model.iteration_index] = np.mean(cost_gradient)
				self.history['cost_gradient']['std'][self.model.iteration_index] = np.std(cost_gradient)
				self.history['cost_gradient']['max'][self.model.iteration_index] = np.max(cost_gradient)
				self.history['cost_gradient']['min'][self.model.iteration_index] = np.min(cost_gradient)
				self.history['cost_gradient']['sum'][self.model.iteration_index] = np.sum(cost_gradient)
				self.history['cost_gradient']['median'][self.model.iteration_index] = np.median(cost_gradient)

		def define_details(self):
			details = {
				'LAYER_INDEX':self.MODEL_STRUCTURE_INDEX,
				'LAYER_TYPE':self.LAYER_TYPE
			}
			if self.LAYER_TYPE is 'CONV':
				details.update({
					'NUM_FILTERS':self.NUM_FILTERS,
					'STRIDE':self.STRIDE
				})
			elif self.LAYER_TYPE is 'POOL':
				details.update({
					'STRIDE':self.STRIDE,
					'POOL_TYPE':self.POOL_TYPE
				})
			elif self.LAYER_TYPE is 'FLATTEN':
				details.update({
				})
			elif self.LAYER_TYPE is 'FC':
				details.update({
					'NUM_NODES':self.NUM_NODES,
					'ACTIVATION':self.ACTIVATION
				})
			elif self.LAYER_TYPE is 'ACTIVATION':
				details.update({
					'FUNCTION':self.FUNCTION
				})
			
			return details

		def _initiate_adam_params(self):
			if self.LAYER_TYPE == 'FC':
				self.adam_params = {
					'weight':{
						'moment1':np.zeros(shape=self.weights.shape),
						'moment2':np.zeros(shape=self.weights.shape)
					},
					'bias':{
						'moment1':np.zeros(shape=self.bias.shape),
						'moment2':np.zeros(shape=self.bias.shape)
					},
					'epsilon':1e-8
				}
			else:
				self.adam_params = {
					'filter':{
						'moment1':np.zeros(shape=self.filters.shape),
						'moment2':np.zeros(shape=self.filters.shape)
					},
					'bias':{
						'moment1':np.zeros(shape=self.bias.shape),
						'moment2':np.zeros(shape=self.bias.shape)
					},
					'epsilon':1e-8
				}

		def _update_factor(self,cost_gradient,param_type=None):
			if self.model.OPTIMISER_METHOD == 'adam':
				assert param_type in ('weight','filter','bias')
				moment1 = self.adam_params[param_type]['moment1']
				moment2 = self.adam_params[param_type]['moment2']
				eps = self.adam_params['epsilon']

				moment1 = self.model.BETA1 * moment1 + (1 - self.model.BETA1) * cost_gradient
				moment2 = self.model.BETA2 * moment2 + (1 - self.model.BETA2) * np.square(cost_gradient)
				moment1_hat = moment1 / (1 - np.power(self.model.BETA1,self.model.iteration_index + 1))
				moment2_hat = moment2 / (1 - np.power(self.model.BETA2,self.model.iteration_index + 1))

				self.adam_params[param_type]['moment1'] = moment1
				self.adam_params[param_type]['moment2'] = moment2
				return self.model.LEARNING_RATE * ( moment1_hat / np.sqrt( moment2_hat + eps ) )
			else:
				return self.model.LEARNING_RATE * cost_gradient


	class Conv_Layer(CNN_Layer):
		def __init__(self,filt_shape: tuple,num_filters: int=5,stride: int=1,padding: int=0,pad_type: str=None,random_seed=42,initiation_method=None):
			""" 
			- filt_shape (tuple): A tuple object describing the 2D shape of the filter to be convolved over the input.
			- num_filters (int): Number of filters to be used.
			- stride (int): Size of steps to take when shifting the filter. (Currently stride_x = stride_y).
			- padding (int): Width of zero-padding to apply on each side of the array. Only applied if pad_type is None.
			- pad_type (str): Options: same (output shape is same as input shape), valid (equal to padding=0), include (padding added evenly on all sides of the array to allow the filter to shift over the input an integer number of times - avoid excluding input data).
			- random_seed (int): The seed provided to numpy before initiating the filters and biases. random_seed=None will result in no seed being provided meaning numpy will generate it dynamically each time.
			- initiation_method (str): This is the method used to initiate the weights and biases. Options: "kaiming", "xavier" or None. Default is none - this simply takes random numbers from standard normal distribution with no scaling.
			"""
			super().__init__()

			self.LAYER_TYPE = 'CONV'
			self.IS_TRAINABLE = True
			self.FILT_SHAPE = filt_shape	# 2D tuple describing num rows and cols
			self.NUM_FILTERS = num_filters
			self.STRIDE = stride
			self.PADDING = padding
			self.PAD_TYPE = None if pad_type is None else pad_type.lower()
			self.RANDOM_SEED = random_seed
			self.INITIATION_METHOD = None if initiation_method is None else initiation_method.lower()

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
			self.filters = CNN.array_init(shape=(self.NUM_FILTERS,INPUT_SHAPE[0],self.FILT_SHAPE[0],self.FILT_SHAPE[1]),method=self.INITIATION_METHOD)
			np.random.seed(self.RANDOM_SEED)
			self.bias = np.zeros(shape=(self.NUM_FILTERS,1))

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
					if (NUM_INPUT_ROWS - self.FILT_SHAPE[0]) % self.STRIDE != 0:
						pad_rows_needed = self.FILT_SHAPE[0] - ((NUM_INPUT_ROWS - self.FILT_SHAPE[0]) % self.STRIDE)
					else:
						pad_rows_needed = 0
					if (NUM_INPUT_COLS - self.FILT_SHAPE[1]) % self.STRIDE != 0:
						pad_cols_needed = self.FILT_SHAPE[1] - ((NUM_INPUT_COLS - self.FILT_SHAPE[1]) % self.STRIDE)
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

			if self.model.OPTIMISER_METHOD == 'adam': self._initiate_adam_params()

		def _forwards(self,_input):
			self.output[:] = 0 # Output must be re-initiated before each run

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

			self._track_metrics(output=self.output)
			return self.output	# NOTE: Output is 3D array of shape: ( NUM_FILTS, NUM_ROWS, NUM_COLS )

		def _backwards(self,cost_gradient):	
			assert cost_gradient.shape == self.output.shape, f'cost_gradient shape [{cost_gradient.shape}] does not match layer output shape [{self.output.shape}].'
			self._track_metrics(cost_gradient=cost_gradient)

			_, c_rows, c_cols = cost_gradient.shape
			dilation_idx_row = np.arange(c_rows-1) + 1	# Intiatial indices for insertion of zeros
			dilation_idx_col = np.arange(c_cols-1) + 1	# Intiatial indices for insertion of zeros
			cost_gradient_dilated = cost_gradient
			for n in range(1,self.STRIDE):
				cost_gradient_dilated = np.insert(
					np.insert( cost_gradient_dilated, dilation_idx_row * n, 0, axis=1 ),
					dilation_idx_col * n, 0, axis=2)	# the n multiplier is to increment the indices in the non-uniform manner required.
			# print(f'cost_gradient shape: {cost_gradient.shape} | cost_gradient_dilated shape: {cost_gradient_dilated.shape}')

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
			assert dCdF.shape == self.filters.shape, f'dCdF shape [{dCdF.shape}] does not match filters shape [{self.filters.shape}].'
			
			dCdX_pad = np.zeros(shape=self.padded_input.shape)
			dCdX_pad[:,: dCdX_pad.shape[1] - pxls_excl_y, : dCdX_pad.shape[2] - pxls_excl_x] = dCdX_pad_excl	# pixels excluded in forwards pass will now appear with cost_gradient = 0.

			# Remove padding that was added to the input array.
			dCdX = dCdX_pad[ : , self.ROW_UP_PAD : dCdX_pad.shape[1] - self.ROW_DOWN_PAD , self.COL_LEFT_PAD : dCdX_pad.shape[2] - self.COL_RIGHT_PAD ]
			assert dCdX.shape == self.input.shape, f'dCdX shape [{dCdX.shape}] does not match layer input shape [{self.input.shape}].'

			# ADJUST THE FILTERS
			# self.filters = self.filters - ( self.model.LEARNING_RATE * dCdF	)
			self.filters = self.filters - self._update_factor(dCdF,'filter')

			# ADJUST THE BIAS
			dCdB = 1 * cost_gradient.sum(axis=(1,2)).reshape(self.bias.shape)
			assert dCdB.shape == self.bias.shape, f'dCdB shape [{dCdB.shape}] does not match bias shape [{self.bias.shape}].'
			# self.bias = self.bias - ( self.model.LEARNING_RATE * dCdB )	# NOTE: Adjustments done in opposite direction to cost_gradient
			self.bias = self.bias - self._update_factor(dCdB,'bias')	# NOTE: Adjustments done in opposite direction to cost_gradient

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

	
	class Pool_Layer(CNN_Layer):
		def __init__(self,filt_shape: tuple,stride: int,pool_type: str='max',padding: int=0,pad_type: str=None):
			'''
			- filt_shape (tuple): A tuple object describing the 2D shape of the filter to use for pooling.
			- stride (int): Size of steps to take when shifting the filter. (Currently stride_x = stride_y).
			- pool_type (str): Pooling method to be applied. Options = max, min, mean.
			- padding (int): Width of zero-padding to apply on each side of the array. Only applied if pad_type is None.
			- pad_type (str): Options: same (output shape is same as input shape), valid (equal to padding=0), include (padding added evenly on all sides of the array to allow the filter to shift over the input an integer number of times - avoid excluding input data).
			'''
			super().__init__()

			self.LAYER_TYPE = 'POOL'
			self.IS_TRAINABLE = False
			self.FILT_SHAPE = filt_shape	# 2D array (rows,cols)
			self.STRIDE = stride
			self.POOL_TYPE = pool_type.lower()
			assert self.POOL_TYPE in ('max','mean','min')
			self.PADDING = padding
			self.PAD_TYPE = None if pad_type is None else pad_type.lower()

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
					if (NUM_INPUT_ROWS - self.FILT_SHAPE[0]) % self.STRIDE != 0:
						pad_rows_needed = self.FILT_SHAPE[0] - ((NUM_INPUT_ROWS - self.FILT_SHAPE[0]) % self.STRIDE)
					else:
						pad_rows_needed = 0
					if (NUM_INPUT_COLS - self.FILT_SHAPE[1]) % self.STRIDE != 0:
						pad_cols_needed = self.FILT_SHAPE[1] - ((NUM_INPUT_COLS - self.FILT_SHAPE[1]) % self.STRIDE)
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

			self._track_metrics(output=self.output)
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
			self._track_metrics(cost_gradient=cost_gradient)
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


	class Flatten_Layer(CNN_Layer):
		""" A psuedo layer that simply adjusts the data dimension as it passes between 2D/3D Conv or Pool layers to the 1D FC layers. """

		def __init__(self):
			super().__init__()

			self.LAYER_TYPE = 'FLATTEN'
			self.IS_TRAINABLE = False

		def prepare_layer(self):
			if self.prev_layer is None:
				self.output = np.zeros(shape=(np.prod(self.model.INPUT_SHAPE),1))
			else:
				self.output = np.zeros(shape=(self.prev_layer.output.size,1))

		def _forwards(self,_input):
			self.output = _input.reshape((_input.size,1))

			self._track_metrics(output=self.output)
			return self.output	# NOTE: Vertical array

		def _backwards(self,cost_gradient):
			self._track_metrics(cost_gradient=cost_gradient)
			if self.prev_layer is not None:
				return cost_gradient.reshape(self.prev_layer.output.shape)


	class FC_Layer(CNN_Layer):
		"""
		The Fully Connected Layer is defined as being the layer of nodes and the weights of the connections that link those nodes to the previous layer.
		"""
		def __init__(self, num_nodes, activation: str=None,random_seed=42,initiation_method=None):
			"""
			- n: Number of nodes in layer.
			- activation: The name of the activation function to be used. The activation is handled by a CNN.Activation_Layer object that is transparent to the user here. Defaults to None - a transparent Activation layer will still be added however, the data passing through will be untouched.
			- initiation_method (str): This is the method used to initiate the weights and biases. Options: "kaiming", "xavier" or None. Default is none - this simply takes random numbers from standard normal distribution with no scaling.
			"""
			super().__init__()

			self.LAYER_TYPE = 'FC'
			self.IS_TRAINABLE = True
			self.NUM_NODES = num_nodes
			self.ACTIVATION = None if activation is None else activation.lower()
			self.RANDOM_SEED = random_seed
			self.INITIATION_METHOD = None if initiation_method is None else initiation_method.lower()

		def prepare_layer(self):
			if self.prev_layer is None:
				w_cols = np.prod(self.model.INPUT_SHAPE)
			else:
				w_cols = self.prev_layer.output.size
			
			w_rows = self.NUM_NODES	# NOTE: "Each row corresponds to all connections of previous layer to a single node in current layer." - based on vertical node array.
			np.random.seed(self.RANDOM_SEED)
			# self.weights = np.random.normal(size=(w_rows,w_cols))	# NOTE: this is the correct orientation for vertical node array.
			self.weights = CNN.array_init(shape=(w_rows,w_cols),method=self.INITIATION_METHOD)	# NOTE: this is the correct orientation for vertical node array.

			self.bias = np.zeros(shape=(self.NUM_NODES,1))	# NOTE: Recommended to initaite biases to zero.
			
			self.output = np.zeros(shape=(self.NUM_NODES,1))	# NOTE: This is a vertical array.

			if self.model.OPTIMISER_METHOD == 'adam': self._initiate_adam_params()

		def _forwards(self,_input):
			self.input = _input.reshape((-1,1))	# Convert to vertical

			self.output = np.dot( self.weights, self.input ) + self.bias
			
			assert self.output.shape == (self.NUM_NODES,1)
			self._track_metrics(output=self.output)
			print(f'Layer: {self.MODEL_STRUCTURE_INDEX} output:',self.output)
			return self.output

		def _backwards(self, cost_gradient):
			"""
			Take cost gradient dC/dZ (how the output of this layer affects the cost) and backpropogate

			Z = W . I + B

			"""
			assert cost_gradient.shape == self.output.shape
			self._track_metrics(cost_gradient=cost_gradient)

			dZ_dW = np.transpose( self.input )	# Partial diff of weighted sum (Z) w.r.t. weights
			dZ_dB = 1
			dZ_dI = np.transpose( self.weights )	# Partial diff of weighted sum w.r.t. input to layer.
			
			dC_dW = np.multiply( cost_gradient , dZ_dW )	# Element-wise multiplication. The local gradient needs transposing for the multiplication.
			assert dC_dW.shape == self.weights.shape, f'dC_dW shape {dC_dW.shape} does not match {self.weights.shape}'
			# self.weights = self.weights - ( self.model.LEARNING_RATE * dC_dW )	# NOTE: Adjustments done in opposite direction to cost_gradient
			self.weights = self.weights - self._update_factor(dC_dW,'weight')

			dC_dB = np.multiply( cost_gradient, dZ_dB )	# Element-wise multiplication

			assert dC_dB.shape == self.bias.shape, f'dC_dW shape {dC_dB.shape} does not match {self.bias.shape}'
			# self.bias = self.bias - ( self.model.LEARNING_RATE * dC_dB )	# NOTE: Adjustments done in opposite direction to cost_gradient
			self.bias = self.bias - self._update_factor(dC_dB,'bias')	# NOTE: Adjustments done in opposite direction to cost_gradient

			return np.matmul( dZ_dI , cost_gradient )	# Matrix multiplication


	class Activation(CNN_Layer):
		def __init__(self,function: str=None,alpha=0.01):
			super().__init__()

			self.LAYER_TYPE = 'ACTIVATION'
			self.IS_TRAINABLE = False
			self.alpha = alpha

			self.FUNCTION = None if function is None else function.lower()

		def prepare_layer(self):
			self.output = np.zeros(shape=self.prev_layer.output.shape )

		def _forwards(self,_input):
			self.input = _input
			
			if self.FUNCTION is None:
				self.output = _input
			elif self.FUNCTION == 'relu':
				# The ReLu function is highly computationally efficient but is not able to process inputs that approach zero or negative.
				_input[_input<0] = 0
				self.output = _input
			elif self.FUNCTION == 'softmax':
				# Softmax is a special activation function used for output neurons. It normalizes outputs for each class between 0 and 1, and returns the probability that the input belongs to a specific class.
				exp = np.exp(_input - np.max(_input))	# Normalises by max value - provides "numerical stability"
				self.output = exp / np.sum(exp)
				# print(_input)
				# print(self.output)
				assert round(self.output.sum()) == 1, f'Output array sum {self.output.sum()} is not equal to 1.\nInput Array: {self.input.reshape((1,-1))}\nOuput Array: {self.output.reshape((1,-1))}'
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
				_input[_input <= 0] = self.alpha * _input[_input <= 0]
				self.output = _input
			elif self.FUNCTION == 'parametric relu': # TODO: Define "Parametric ReLu"
				#  The Parametric ReLu function allows the negative slope to be learned, performing backpropagation to learn the most effective slope for zero and negative input values.
				pass

			self._track_metrics(output=self.output)
			print(f'Layer: {self.MODEL_STRUCTURE_INDEX} output:',self.output)
			return self.output

		def _backwards(self,cost_gradient):
			# NOTE: Derivative of Activation w.r.t. z
			self._track_metrics(cost_gradient=cost_gradient)

			dA_dZ = None
			if self.FUNCTION is None: # a = z
				dA_dZ = np.ones( self.input.shape )
			elif self.FUNCTION == 'relu':
				dA_dZ = self.input
				dA_dZ[dA_dZ <= 0] = 0
				dA_dZ[dA_dZ > 0] = 1
			elif self.FUNCTION == 'softmax':
				# Generalised function is: del_sig(zi) / del_zj = sig(z_j)(d_kron - sig(z_i)). Where d_kron is Kronecker delta: 1 if i=j and 0 otherwise.
				sig_square = np.broadcast_to( self.output ,(self.output.shape[-2],self.output.shape[-2]))	# Broadcasting output array to a square
				identity = np.identity(self.output.shape[-2])	# Square identity array
				sig_t_square = np.transpose(sig_square)
				# dA_dZ = np.sum(sig_square * (identity - sig_t_square), axis=1, keepdims=True)	# Sum along rows -> result in vertical array
				dA_dZ = sig_square * (identity - sig_t_square)	# Sum along rows -> result in vertical array

				cost_gradient = np.broadcast_to( np.transpose(cost_gradient) , (cost_gradient.shape[-2],cost_gradient.shape[-2]))
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
			
			cost_gradient = np.sum( np.multiply( dA_dZ , cost_gradient ), axis=1, keepdims=True)	# Element-wise multiplication. Sum in case square matrix.
			assert cost_gradient.shape == self.input.shape
			return cost_gradient

