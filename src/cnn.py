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

	def __init__(self,optimiser_method='gd'):
		'''
		- optimiser_method (str): Options: ('gd','momentum','rmsprop','adam'). Default is 'gd'.
		'''
		assert optimiser_method.lower() in CNN.SUPPORTED_OPTIMISERS, f'You must provide an optimiser that is supported. The options are: {CNN.SUPPORTED_OPTIMISERS}'

		self.is_prepared = False

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
				print('--> Expected output shape:',curr_layer.OUTPUT_SHAPE)
				if curr_layer.MODEL_STRUCTURE_INDEX == 0:
					# First layer; set model input shape.
					self.INPUT_SHAPE = curr_layer.INPUT_SHAPE

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

	def train(self,Xs,ys,epochs,max_batch_size=32,shuffle=False,random_seed=42,learning_rate=0.01,cost_fn='mse',beta1=0.9,beta2=0.999):
		'''
		Should take array of inputs and array of labels of the same length.

		[For each epoch] For each input, propogate forwards and backwards.

		ARGS:
		- Xs (np.ndarray or list): (N,ch,rows,cols) or (N,features). Where N is number of examples, ch is number of channels.
		- ys (np.ndarray or list): (N,num_categories). e.g. ex1 label = [0,0,0,1,0] for 5 categories (one-hot encoded).
		- epochs (int): Number of iterations over the data set.
		- max_batch_size (int): Maximum number of examples in each batch. (Tensorflow defaults to 32. Also, batch_size > N will be truncated.)
		- shuffle (bool): Determines whether the data set is shuffled before training.
		- random_seed (int): The seed provided to numpy before performing the shuffling. random_seed=None will result in no seed being provided meaning numpy will generate it dynamically each time.
		- beta1 (float): param used for Adam optimisation
		- beta2 (float): param used for Adam optimisation
		'''
		Xs, ys = np.array(Xs), np.array(ys)	# Convert data to numpy arrays in case not already.
		ys = ys.reshape(-1,1) if ys.ndim == 1 else ys
		# --------- ASSERTIONS -----------
		# Check shapes and orientation are as expected
		assert self.structure[-1].LAYER_TYPE in ('FC','ACTIVATION'), 'Model must have either FC or ACTIVATION as final layer.'
		assert Xs.shape[0] == ys.shape[0], f'Dimension (0) of input data [{Xs.shape}] and labels [{ys.shape}] does not match.'
		assert Xs.ndim in (2,4), 'Xs must be either 2 dimensions (for NN) or 4 dimensions (for CNN).'
		if Xs.ndim == 4:
			assert Xs.shape[1:] == self.INPUT_SHAPE, f'Expected X shape to be: {self.INPUT_SHAPE}, instead received: {Xs.shape[1:]}'
		elif Xs.ndim == 2:
			assert (Xs.shape[1],1) == self.INPUT_SHAPE, f'Expected X shape to be: {self.INPUT_SHAPE}, instead received: {(Xs.shape[1],1)}'
		assert ys.ndim == 2, f'ys should be a 2 dimensional array.'
		assert type(epochs) == int, 'An integer value must be supplied for argument "epochs"'
		assert int(max_batch_size) == max_batch_size and max_batch_size is not None, 'An integer value must be supplied for argument "max_batch_size"'
		assert cost_fn.lower() in CNN.SUPPORTED_COST_FUNCTIONS, f'Chosen cost function not supported, please choose: {CNN.SUPPORTED_COST_FUNCTIONS}'
		
		
		# --------- ASSIGNMENTS ----------
		self.N = Xs.shape[0]	# Total number of examples in Xs
		(self.Xs, self.ys) = CNN.shuffle(Xs,ys,random_seed) if shuffle else (Xs, ys)
		self.EPOCHS = epochs
		self.MAX_BATCH_SIZE = self.N if (max_batch_size is None or max_batch_size > self.N) else max_batch_size
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
		
		# # Orientate the labels to be (nx,m)
		# if self.ys.shape[1] == self.structure[-1].OUTPUT_SHAPE[0] and self.ys.shape[0] != self.structure[-1].OUTPUT_SHAPE[0]:
			# self.ys = self.ys.T.copy()

		# ---------- TRAIN -------------
		train_start = dt.now()
		for epoch_ind in range(self.EPOCHS):
			self.epoch_ind = epoch_ind
			self.epoch_accuracy = 0
			
			self._iterate_forwards()

		return dt.now(), dt.now() - train_start	# returns training finish time and duration.

	def print_train_progress(self,batch_index):
		progess_bar_length = 30	# characters (not including '[' ']')
		progress = (batch_index+1) / self.BATCH_COUNT
		progressor = '=' * int(progress * progess_bar_length)
		if progress < 1:
			progressor += '>'
		
		progress_string = '[' + progressor + '-' * (progess_bar_length - len(progressor)) + ']'

		metrics_string = f'Batch: {batch_index+1}/{self.BATCH_COUNT} | Acc: {self.epoch_accuracy*100:.2f}%'

		print_string = f'Epoch {self.epoch_ind + 1}/{self.EPOCHS} ' + progress_string + ' ' + metrics_string

		if batch_index + 1 == self.BATCH_COUNT:
			print(print_string,end='\n')
		else:
			print(print_string,end='\r')

	SUPPORTED_OPTIMISERS = ('gd','momentum','rmsprop','adam')

	def _iterate_forwards(self):
		for batch_ind in range(self.BATCH_COUNT):
			ind_lower = batch_ind * self.MAX_BATCH_SIZE	# Lower bound of index range
			ind_upper = batch_ind * self.MAX_BATCH_SIZE + self.MAX_BATCH_SIZE	# Upper bound of index range
			if ind_upper > self.N and self.N > 1:
				ind_upper = self.N
			self.current_batch_size = ind_upper - ind_lower

			# print('Lower index:',ind_lower,'Upper index:',ind_upper)
			# print(self.Xs)
			# print(self.BATCH_COUNT, self.Xs.shape)
			batch_Xs = self.Xs[ ind_lower : ind_upper ].copy()
			batch_ys = self.ys[ ind_lower : ind_upper ].copy()
			# print(batch_Xs.shape,batch_ys.shape)

			predictions = self.predict(batch_Xs,training=True)

			self.iteration_cost = self.cost(predictions, batch_ys)
			self.iteration_cost_gradient = self.cost(predictions,batch_ys,derivative=True)

			batch_correct = np.sum((np.argmax(batch_ys.T,axis=0) == np.argmax(predictions,axis=0)))
			self.epoch_accuracy = (self.epoch_accuracy * ind_lower + batch_correct) / (ind_upper+1)
			# for ex_ind , X in enumerate(batch_Xs):	# For each example (observation)
			# 	print(X.shape)
			# 	prediction = self.predict(X,training=True)

			# 	self.iteration_cost += self.cost(prediction, batch_ys[ex_ind],batch_size=batch_size)
			# 	self.iteration_cost_gradient += self.cost(prediction, batch_ys[ex_ind],batch_size=batch_size,derivative=True)

			# print(f'-- Epoch: {self.epoch_ind+1}/{self.EPOCHS } | Batch: {batch_ind+1}/{self.BATCH_COUNT} | Cost: {self.iteration_cost}')
			self.print_train_progress(batch_ind)

			self._iterate_backwards()

	def _iterate_backwards(self):
		self.iteration_index += 1
		self.history['cost'][self.iteration_index] = self.iteration_cost
		# Backpropagate the cost_gradient
		cost_gradient = self.iteration_cost_gradient
		for layer in self.structure[::-1]:
			cost_gradient = layer._backwards(cost_gradient)

		self.iteration_cost = 0
		self.iteration_cost_gradient = 0

	def predict(self,Xs,training=False):
		if training: self.feed_forwards_cycle_index += 1
		for layer in self.structure:
			Xs = layer._forwards(Xs)
			# print('Layer index:',layer.MODEL_STRUCTURE_INDEX)
			# print('Output:',X)
		return Xs

	def evaluate(self,Xs,ys):
		predictions = self.predict(Xs,training=False)
		accuracy = np.sum((np.argmax(ys.T,axis=0) == np.argmax(predictions,axis=0))) / len(Xs)
		return accuracy

	@staticmethod
	def shuffle(X,y,random_seed=None):
		if random_seed is not None:
			np.random.seed(random_seed)
		permutation = np.random.permutation( X.shape[0] )
		X_shuffled = X[permutation]
		y_shuffled = y[permutation]
		print(X_shuffled.shape,y_shuffled.shape)
		assert X.shape == X_shuffled.shape, f'X shape: {X.shape} | X shuffled shape: {X_shuffled.shape}'
		return (X_shuffled, y_shuffled)

	def _initiate_tracking_metrics(self):
		# Initiate model history tracker
		initial_arr = np.zeros(self.TOTAL_ITERATIONS)
		initial_arr[:] = np.NaN
		self.history = {'cost':initial_arr}	# dict object will allow us to store history of various parameters.
		
		# Initiate layer history trackers
		for layer in self.structure:
			layer._initiate_history()

	SUPPORTED_COST_FUNCTIONS = ('mse','cross_entropy')

	def cost(self,predictions,labels,derivative=False):
		'''
		Cost function to provide measure of model 'correctness'. returns vector cost value.
		'''
		labels = labels.T	# Transpose labels to (cats,batch_size)
		assert labels.ndim == 2, f'Expected 2 dimensional array; instead got {labels.ndim} dims.'
		assert predictions.shape == labels.shape, f'Model output shape, {predictions.shape}, does not match labels shape, {labels.shape}.'
		batch_size = labels.shape[1]

		if self.COST_FN == 'mse':
			error = predictions - labels	# Vector
			if not derivative:
				return np.sum( np.square( error ) ) / (batch_size * labels.shape[0])	# Vector
			else:
				return -( 2 * error ) / batch_size	# Vector
		elif self.COST_FN == 'cross_entropy':
			if not derivative:
				# print('logprobs:',np.log(predictions))
				cost = -np.sum(labels * np.log(predictions)) / batch_size
				# print('Cost:',cost)
				return cost
			else:
				return - np.divide(labels,predictions) / batch_size

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
	def array_init(shape,method=None,seed=None):
		''' Random initialisation of weights array.
		Xavier or Kaiming: (https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79) '''
		assert len(shape) >= 2
		fan_in = shape[-1]
		fan_out = shape[-2]

		if seed:
			np.random.seed(seed)

		if method is None:
			array = np.random.randn(fan_out,fan_in) * 0.01
		elif method == 'kaiming_normal':
			# AKA "he normal" after Kaiming He.
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
		else:
			raise BaseException('ERROR: Unrecognised array initialisation method: ' + method)

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
		def __init__(self,filt_shape: tuple or int,num_filters: int=5,stride: int=1,padding: int=0,pad_type: str=None,random_seed=42,initiation_method=None,input_shape=None):
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
			super().__init__()

			self.LAYER_TYPE = 'CONV'
			self.IS_TRAINABLE = True
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

			# Initiate filters
			self.filters = CNN.array_init(shape=(self.NUM_FILTERS,self.INPUT_SHAPE[0],self.FILT_SHAPE[0],self.FILT_SHAPE[1]),method=self.INITIATION_METHOD,seed=self.RANDOM_SEED)
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

			self.OUTPUT_SHAPE = (self.NUM_FILTERS,row_out,col_out)
			# self.output = np.zeros(shape=(self.NUM_FILTERS,row_out,col_out))	# Output initiated.
			if self.PAD_TYPE == 'same':
				assert self.OUTPUT_SHAPE[-2:] == self.INPUT_SHAPE[-2:]	# Channels may differ.

			if self.model.OPTIMISER_METHOD == 'adam': self._initiate_adam_params()

		def _forwards(self,_input):
			# self.output[:] = 0 # Output must be re-initiated before each run
			# if _input.ndim == 3:
			# 	self.input = _input
			# elif _input.ndim == 2:
			# 	self.input = np.array( [ _input ] )	# NOTE: 'fakes' number of channels to be 1.
			assert _input.ndim == 4 and _input.shape[1:] == self.INPUT_SHAPE, f'Input shape, {_input.shape[1:]}, expected to be, {self.INPUT_SHAPE} for each example (observation).'

			batch_size = _input.shape[0]

			# Apply the padding to the input.
			self.padded_input = np.pad(self.input,[(0,0),(0,0),(self.ROW_UP_PAD,self.ROW_DOWN_PAD),(self.COL_LEFT_PAD,self.COL_RIGHT_PAD)],'constant',constant_values=(0,0))

			self.output = np.zeros(shape=(batch_size,*self.OUTPUT_SHAPE))

			# proc_rows, proc_cols = self.padded_input.shape[-2:]
			for i in range(batch_size):
				for filt_index in range(self.NUM_FILTERS):
					filt = self.filters[filt_index]
					
					filt_channels, filt_rows, filt_cols = filt.shape

					for channel_index in range(filt_channels):
						self.output[i,filt_index] += CNN.Conv_Layer.convolve( self.padded_input[i,channel_index], filt[channel_index], self.STRIDE )
					
					self.output[i,filt_index] += self.bias[filt_index]


			self._track_metrics(output=self.output)
			return self.output	# NOTE: Output is 3D array of shape: ( NUM_FILTS, NUM_ROWS, NUM_COLS )

		def _backwards(self,cost_gradient):	
			assert cost_gradient.shape == self.output.shape, f'cost_gradient shape [{cost_gradient.shape}] does not match layer output shape [{self.output.shape}].'
			self._track_metrics(cost_gradient=cost_gradient)

			_,_, c_rows, c_cols = cost_gradient.shape
			dilation_idx_row = np.arange(c_rows-1) + 1	# Intiatial indices for insertion of zeros
			dilation_idx_col = np.arange(c_cols-1) + 1	# Intiatial indices for insertion of zeros
			for n in range(1,self.STRIDE):
				cost_gradient_dilated = np.insert(
					np.insert( cost_gradient, dilation_idx_row * n, 0, axis=2 ),
					dilation_idx_col * n, 0, axis=3)	# the n multiplier is to increment the indices in the non-uniform manner required.
			# print(f'cost_gradient shape: {cost_gradient.shape} | cost_gradient_dilated shape: {cost_gradient_dilated.shape}')
			# assert cost_gradient_dilated.shape == self.input.shape, f'Dilated cost gradient shape, {cost_gradient_dilated.shape}, does not match layer input shape, {self.input.shape}.'

			batch_size, channels, height, width = self.padded_input.shape

			# Account for filter not shifting over input an integer number of times with given stride.
			pxls_excl_x = (self.padded_input.shape[3] - self.FILT_SHAPE[1]) % self.STRIDE	# pixels excluded in x direction (cols)
			pxls_excl_y = (self.padded_input.shape[2] - self.FILT_SHAPE[0]) % self.STRIDE	# pixels excluded in y direction (rows)

			# dCdF = []	# initiate as list then convert to np.array
			# dCdX_pad_excl = []
			dCdF = np.zeros(shape=self.filters.shape)
			dCdX_pad = np.zeros(shape=self.padded_input.shape)
			# Find cost gradient wrt previous output and filters.
			rotated_filters = np.rot90( self.filters, k=2, axes=(1,2) )	# rotate 2x90 degs, rotating in direction of rows to columns.
			for i in range(batch_size):
				for filt_index in range(self.NUM_FILTERS):
					# filt_1_container = []
					# filt_2_container = []
					for channel_index in range(channels):
						# dCdF
						# filt_1_container.append( CNN.Conv_Layer.convolve( self.padded_input[channel_index], cost_gradient_dilated[r_filt_ind], stride=1 ) )
						dCdF[filt_index, channel_index] += CNN.Conv_Layer.convolve( self.padded_input[i,channel_index], cost_gradient_dilated[i,filt_index], stride=1 )
						# dCdX
						# filt_2_container.append( CNN.Conv_Layer.convolve( cost_gradient_dilated[r_filt_ind], rotated_filters[r_filt_ind][channel_index], stride=1, full_convolve=True ) )
						dCdX_pad[i,channel_index, : dCdX_pad.shape[1] - pxls_excl_y, : dCdX_pad.shape[2] - pxls_excl_x] += CNN.Conv_Layer.convolve( cost_gradient_dilated[i,filt_index], rotated_filters[filt_index,channel_index], stride=1, full_convolve=True )
					# dCdF.append(filt_1_container)
					# dCdX_pad_excl.append(filt_2_container)
			# dCdF = np.array( dCdF )
			# dCdX_pad_excl = np.array( dCdX_pad_excl ).sum(axis=0)	# NOTE: This is the cost gradient w.r.t. the padded input and potentially excluding pixels.
			
			dCdF = dCdF[:,:, : dCdF.shape[2] - pxls_excl_y, : dCdF.shape[3] - pxls_excl_x]	# Remove the values from right and bottom of array (this is where the excluded pixels will be).
			assert dCdF.shape == self.filters.shape, f'dCdF shape [{dCdF.shape}] does not match filters shape [{self.filters.shape}].'
			
			# ADJUST THE FILTERS
			# self.filters = self.filters - ( self.model.LEARNING_RATE * dCdF	)
			self.filters = self.filters - self._update_factor(dCdF,'filter')

			# ADJUST THE BIAS
			assert dCdB.shape == self.bias.shape, f'dCdB shape [{dCdB.shape}] does not match bias shape [{self.bias.shape}].'
			dCdB = 1 * cost_gradient.sum(axis=(1,2)).reshape(self.bias.shape)
			self.bias = self.bias - self._update_factor(dCdB,'bias')	# NOTE: Adjustments done in opposite direction to cost_gradient

			# dCdX_pad = np.zeros(shape=self.padded_input.shape)
			# dCdX_pad[:,: dCdX_pad.shape[1] - pxls_excl_y, : dCdX_pad.shape[2] - pxls_excl_x] = dCdX_pad_excl	# pixels excluded in forwards pass will now appear with cost_gradient = 0.

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

	
	class Pool_Layer(CNN_Layer):
		def __init__(self,filt_shape: tuple or int,stride: int,pool_type: str='max',padding: int=0,pad_type: str=None,input_shape=None):
			'''
			- filt_shape (int/tuple): Number of rows and columns of each filter. INT if rows == cols. TUPLE if rows != cols.
			- stride (int): Size of steps to take when shifting the filter. (Currently stride_x = stride_y).
			- pool_type (str): Pooling method to be applied. Options = max, min, mean.
			- padding (int): Width of zero-padding to apply on each side of the array. Only applied if pad_type is None.
			- pad_type (str): Options: same (output shape is same as input shape), valid (equal to padding=0), include (padding added evenly on all sides of the array to allow the filter to shift over the input an integer number of times - avoid excluding input data).
			- input_shape (tuple): Input shape of a single example (observation). Expected (channels, rows, cols)
			'''
			super().__init__()

			self.LAYER_TYPE = 'POOL'
			self.IS_TRAINABLE = False
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

		def prepare_layer(self):
			""" This needs to be done after the input has been identified - currently happens when train() is called. """
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

			self.OUTPUT_SHAPE = (self.INPUT_SHAPE[0],row_out,col_out)
			# self.output = np.zeros(shape=(self.INPUT_SHAPE[0],row_out,col_out))	# Output initiated.
			if self.PAD_TYPE == 'same':
				assert self.OUTPUT_SHAPE == self.INPUT_SHAPE	# Channels may differ.

		def _forwards(self,_input):
			# if _input.ndim == 3:
			# 	self.input = _input
			# elif _input.ndim == 2:
			# 	self.input = np.array( [ _input ] )	# NOTE: 'fakes' number of channels to be 1.

			assert _input.ndim == 4 and _input.shape[1:] == self.INPUT_SHAPE, f'Input shape, {_input.shape[1:]}, expected to be, {self.INPUT_SHAPE} for each example (observation).'

			# Apply the padding to the input.
			self.padded_input = np.pad(self.input,[(0,0),(0,0),(self.ROW_UP_PAD,self.ROW_DOWN_PAD),(self.COL_LEFT_PAD,self.COL_RIGHT_PAD)],'constant',constant_values=(0,0))

			self.output = np.zeros(shape=(batch_size,*self.OUTPUT_SHAPE))

			batch_size, channels, proc_rows, proc_cols = self.padded_input.shape

			for i in range(batch_size):
				# Shift 'Filter Window' over the image and perform the downsampling
				curr_y = out_y = 0
				while curr_y <= proc_rows - self.FILT_SHAPE[0]:
					curr_x = out_x = 0
					while curr_x <= proc_cols - self.FILT_SHAPE[1]:
						for channel_index in range(channels):
							if self.POOL_TYPE == 'max':
								sub_arr = self.padded_input[i, channel_index, curr_y : curr_y + self.FILT_SHAPE[0], curr_x : curr_x+ self.FILT_SHAPE[1] ]
								self.output[i,channel_index, out_y, out_x] = np.max( sub_arr )
							elif self.POOL_TYPE == 'min':
								sub_arr = self.padded_input[i, channel_index, curr_y : curr_y + self.FILT_SHAPE[0], curr_x : curr_x+ self.FILT_SHAPE[1] ]
								self.output[i,channel_index, out_y, out_x] = np.min( sub_arr )
							elif self.POOL_TYPE == 'mean':
								sub_arr = self.padded_input[i, channel_index, curr_y : curr_y + self.FILT_SHAPE[0], curr_x : curr_x + self.FILT_SHAPE[1] ]
								self.output[i,channel_index, out_y, out_x] = np.mean( sub_arr )

						curr_x += self.STRIDE
						out_x += 1
					curr_y += self.STRIDE
					out_y += 1

			assert len(self.output.shape) == 4 and self.output.shape[1:] == self.OUTPUT_SHAPE, f'Output shape, {self.output.shape[1:]}, not as expected, {self.OUTPUT_SHAPE}'
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
			dC_dIpad = np.zeros_like(self.padded_input)

			batch_size, channels, padded_rows, padded_cols = dC_dIpad.shape

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


	class Flatten_Layer(CNN_Layer):
		""" A psuedo layer that simply adjusts the data dimension as it passes between 2D Conv or Pool layers to the 1D FC layers. 
		The output shapes of the Conv/ Pool layer is expected to be (m,c,h,w) which is converted to (n,m)."""

		def __init__(self,input_shape):
			"""
			input_shape (tuple): Shape of a single example eg (channels,height,width). Irrespective of batch size. 
			"""

			super().__init__()

			self.LAYER_TYPE = 'FLATTEN'
			self.IS_TRAINABLE = False
			if input_shape is not None:
				assert len(input_shape) == 3, f'ERROR: Expected input_shape to be a tuple of length 3; (channels, height, width).'
			self.INPUT_SHAPE = input_shape

		def prepare_layer(self):
			if self.prev_layer is None:	# This means this is the first layer in the structure, so 'input' is the only thing before.
				assert self.INPUT_SHAPE is not None, 'ERROR: Must define input shape for first layer.'
			else:
				self.INPUT_SHAPE = self.prev_layer.OUTPUT_SHAPE
			self.OUTPUT_SHAPE = (np.prod(self.INPUT_SHAPE),1)	# Output shape for a single example.
			# self.output = np.zeros(shape=(np.prod(self.INPUT_SHAPE[1:]),self.INPUT_SHAPE[0]))

		def _forwards(self,_input):
			assert _input.shape[1:] == self.INPUT_SHAPE, f'ERROR:: Input has unexpected shape: {_input.shape[1:]} | expected: {self.INPUT_SHAPE}'
			self.input = _input
			self.output = _input.T.reshape((-1,_input.shape[0]))	# Taking transpose here puts each example into its own column - number of columns == number of examles.

			self._track_metrics(output=self.output)
			return self.output	# NOTE: 2D matrix, (number of nodes, number of examples in batch)

		def _backwards(self,cost_gradient):
			"""
			cost_gradient expected to have shape (n,m) where n == channels * height * width
			"""
			self._track_metrics(cost_gradient=cost_gradient)
			return cost_gradient.reshape(self.input.T.shape).T	# Outputs shape (m,c,h,w)


	class FC_Layer(CNN_Layer):
		"""
		The Fully Connected Layer is defined as being the layer of nodes and the weights of the connections that link those nodes to the previous layer.
		"""
		def __init__(self, num_nodes, activation: str=None,random_seed=42,initiation_method=None,input_shape=None):
			"""
			- n: Number of nodes in layer.
			- activation: The name of the activation function to be used. The activation is handled by a CNN.Activation_Layer object that is transparent to the user here. Defaults to None - a transparent Activation layer will still be added however, the data passing through will be untouched.
			- initiation_method (str): This is the method used to initiate the weights and biases. Options: "kaiming", "xavier" or None. Default is none - this simply takes random numbers from standard normal distribution with no scaling.
			- input_shape (tuple): Input shape for a single example. (n,1) where n is number of input nodes (features).
			"""
			super().__init__()

			self.LAYER_TYPE = 'FC'
			self.IS_TRAINABLE = True
			self.NUM_NODES = num_nodes
			self.ACTIVATION = None if activation is None else activation.lower()
			self.RANDOM_SEED = random_seed
			self.INITIATION_METHOD = None if initiation_method is None else initiation_method.lower()
			if input_shape is not None:
				assert len(input_shape) == 2 and input_shape[1] == 1, 'Invalid input_shape tuple. Expected (n,1)'
			self.INPUT_SHAPE = input_shape

		def prepare_layer(self):
			if self.prev_layer is None:	# This means this is the first layer in the structure, so 'input' is the only thing before.
				assert self.INPUT_SHAPE is not None, 'ERROR: Must define input shape for first layer.'
			else:
				self.INPUT_SHAPE = self.prev_layer.OUTPUT_SHAPE
			
			self.weights = CNN.array_init(shape=(self.NUM_NODES,self.INPUT_SHAPE[0]),method=self.INITIATION_METHOD,seed=self.RANDOM_SEED)	# NOTE: this is the correct orientation for vertical node array.

			self.bias = np.zeros(shape=(self.NUM_NODES,1))	# NOTE: Recommended to initaite biases to zero.
			
			self.OUTPUT_SHAPE = (self.NUM_NODES,1)
			# self.output = np.zeros(shape=(self.NUM_NODES,1))	# NOTE: This is a vertical array.

			if self.model.OPTIMISER_METHOD == 'adam': self._initiate_adam_params()

		def _forwards(self,_input):
			# print(_input.shape)
			if self.prev_layer is None:
				self.input = _input.T
			else:
				assert len(_input.shape) == 2 and _input.shape[0] == self.INPUT_SHAPE[0], f'Expected input of shape {self.INPUT_SHAPE} instead got {(_input.shape[0],1)}'
				self.input = _input

			self.output = np.dot( self.weights, self.input ) + self.bias
			
			assert len(self.output.shape) == 2 and self.output.shape[0] == self.OUTPUT_SHAPE[0], f'Output shape, {(self.output.shape[0],1)}, not as expected, {self.OUTPUT_SHAPE}'
			self._track_metrics(output=self.output)
			# print(f'Layer: {self.MODEL_STRUCTURE_INDEX} output:',self.output)
			return self.output

		def _backwards(self, dC_dZ):
			"""
			Take cost gradient dC/dZ (how the output of this layer affects the cost) and backpropogate

			Z = W . I + B

			"""
			assert dC_dZ.shape == self.output.shape, f'dC/dZ shape, {dC_dZ.shape}, does not match Z shape, {self.output.shape}.'
			self._track_metrics(cost_gradient=dC_dZ)

			dZ_dW = self.input.T	# Partial diff of weighted sum (Z) w.r.t. weights
			dZ_dB = 1
			dZ_dI = self.weights.T	# Partial diff of weighted sum w.r.t. input to layer.
			
			# dC_dW.shape === W.shape = (n(l),n(l-1)) | dZ_dW.shape = (1,n(l-1))
			# dC_dW = np.multiply( dC_dZ , dZ_dW )	# Element-wise multiplication. The local gradient needs transposing for the multiplication.
			dC_dW = np.dot(dC_dZ,dZ_dW)
			assert dC_dW.shape == self.weights.shape, f'dC/dW shape {dC_dW.shape} does not match W shape {self.weights.shape}'
			# self.weights = self.weights - ( self.model.LEARNING_RATE * dC_dW )	# NOTE: Adjustments done in opposite direction to dC_dZ
			self.weights = self.weights - self._update_factor(dC_dW,'weight')

			dC_dB = np.sum(dC_dZ * dZ_dB, axis=1,keepdims=True)	# Element-wise multiplication (dZ_dB turns out to be just 1)

			assert dC_dB.shape == self.bias.shape, f'dC/dB shape {dC_dB.shape} does not match B shape {self.bias.shape}'
			# self.bias = self.bias - ( self.model.LEARNING_RATE * dC_dB )	# NOTE: Adjustments done in opposite direction to dC_dZ
			self.bias = self.bias - self._update_factor(dC_dB,'bias')	# NOTE: Adjustments done in opposite direction to dC_dZ

			dC_dI = np.dot( dZ_dI , dC_dZ )
			assert dC_dI.shape == self.input.shape, f'dC/dI shape {dC_dI.shape} does not match input shape {self.input.shape}.'
			return dC_dI


	class Activation(CNN_Layer):
		def __init__(self,function: str=None,alpha=0.01,input_shape=None):
			super().__init__()

			self.LAYER_TYPE = 'ACTIVATION'
			self.IS_TRAINABLE = False
			self.alpha = alpha
			self.INPUT_SHAPE = input_shape

			self.FUNCTION = None if function is None else function.lower()

		def prepare_layer(self):
			if self.prev_layer is None:	# This means this is the first layer in the structure, so 'input' is the only thing before.
				assert self.INPUT_SHAPE is not None, 'ERROR: Must define input shape for first layer.'
			else:
				self.INPUT_SHAPE = self.prev_layer.OUTPUT_SHAPE
			self.OUTPUT_SHAPE = self.INPUT_SHAPE
			# self.output = np.zeros(shape=self.INPUT_SHAPE )

		def _forwards(self,_input):
			if self.prev_layer.LAYER_TYPE == 'FC':
				assert len(_input.shape) == 2 and _input.shape[0] == self.INPUT_SHAPE[0], f'Expected input of shape {self.INPUT_SHAPE} instead got {(_input.shape[0],1)}'
			self.input = _input
			
			if self.FUNCTION is None:
				self.output = _input
			elif self.FUNCTION == 'relu':	# NOTE: This would work for Conv activation.
				# The ReLu function is highly computationally efficient but is not able to process inputs that approach zero or negative.
				self.output = np.maximum(_input,0)
			elif self.FUNCTION == 'softmax':
				assert self.prev_layer.LAYER_TYPE == 'FC', 'Softmax activation function is not supported for non-FC inputs.'
				# Softmax is a special activation function used for output neurons. It normalizes outputs for each class between 0 and 1, and returns the probability that the input belongs to a specific class.
				exp = np.exp(_input - np.max(_input,axis=0))	# Normalises by max value - provides "numerical stability"
				self.output = exp / np.sum(exp,axis=0)
				# print(_input)
				# print(self.output)
				# assert round(self.output.sum()) == 1, f'Output array sum {self.output.sum()} is not equal to 1.\nInput Array: {self.input.reshape((1,-1))}\nOuput Array: {self.output.reshape((1,-1))}'
			elif self.FUNCTION == 'sigmoid':	# NOTE: This would work for Conv activation.
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
			
			assert self.output.shape == _input.shape, f'Output shape, {self.output.shape}, not the same as input shape, {_input.shape}.'
			self._track_metrics(output=self.output)
			# print(f'Layer: {self.MODEL_STRUCTURE_INDEX} output:',self.output)
			return self.output

		def _backwards(self,dC_dA):
			"""Compute derivative of Activation w.r.t. Z
			NOTE: CURRENTLY NOT SUPPORTED FOR CONV/POOL LAYERS.
			"""
			assert dC_dA.shape == self.output.shape, f'dC/dA shape, {dC_dA.shape}, not as expected, {self.output.shape}.'
			self._track_metrics(cost_gradient=dC_dA)
			dA_dZ = np.zeros(shape=(self.output.shape[1],self.output.shape[0],self.prev_layer.output.shape[0]))	# TODO: Will need varifying for Conv Activation.
			if self.FUNCTION is None: # a = z
				dA_dZ = np.broadcast_to(np.diag(np.ones(dA_dZ.shape[-1])),( *dA_dZ.shape ))	# '*' unpacks the shape tuple.
			elif self.FUNCTION == 'relu':
				# Insert layer input along dA_dZ diagonals - values > 0 -> 1; values <= 0 -> 0
				ix,iy = np.diag_indices_from(dA_dZ[0,:,:])
				dA_dZ[:,iy,ix] = (self.input.T > 0).astype(int)
			elif self.FUNCTION == 'softmax':
				# Vectorised implementation from https://stackoverflow.com/questions/59286911/vectorized-softmax-gradient
				# NOTE: Transpose is required to create the square matrices of each set of node values.
				outputT = self.output.T
				diag_matrices = outputT.reshape(outputT.shape[0],-1,1) * np.diag(np.ones(outputT.shape[1]))	# Diagonal Matrices
				outer_product = np.matmul(outputT.reshape(outputT.shape[0],-1,1), outputT.reshape(outputT.shape[0],1,-1))	# Outer product
				Jsm = diag_matrices - outer_product
				dA_dZ = Jsm	# NOTE: Even though this equation uses softmax transpose at start, the output does not require transposing because the softmax derivative is symmetrical along diagonal.

			elif self.FUNCTION == 'sigmoid':
				# sig (1 - sig) across diagonals
				ix,iy = np.diag_indices_from(dA_dZ[0,:,:])
				dA_dZ[:,iy,ix] = (self.output * (1 - self.output)).T	# Element-wise multiplication.
			elif self.FUNCTION == 'step': # TODO: Define "step function" derivative
				dA_dZ = None
			elif self.FUNCTION == 'tanh':
				dA_dZ = np.diag((1 - np.square( self.output )).flatten())
			elif self.FUNCTION == 'swish': # TODO: Define "Swish function" derivative
				dA_dZ = None
			elif self.FUNCTION == 'leaky relu':
				ix,iy = np.diag_indices_from(dA_dZ[0,:,:])
				dA_dZ[:,iy,ix] = ( (self.input > 0).astype(int) + ((self.input < 0).astype(int) * self.alpha ) ).T

				# input_diag = np.diag(self.input.flatten())
				# input_diag[input_diag > 0] = 1
				# input_diag[input_diag < 0] = self.alpha
				# dA_dZ = input_diag
			elif self.FUNCTION == 'parametric relu': # TODO: Define "Parametric ReLu" derivative
				dA_dZ = None
			
			assert dA_dZ is not None, f'No derivative defined for chosen activation function "{self.FUNCTION}"'
			assert dA_dZ.shape[1:] == (self.output.shape[0],self.output.shape[0]), 'dA/dZ is expected to be a square matrix (for each example in batch) containing gradient between each activation node and each input node.'
			# print('Layer: ', self.LAYER_TYPE)
			# print('Local gradient shape:',dA_dZ.shape)
			# print('Cost gradient shape:',dC_dA.shape)

			dC_dAexpanded = dC_dA.T.reshape((dC_dA.T.shape[0],-1,1))
			dC_dZexpanded = np.matmul(dA_dZ,dC_dAexpanded)
			dC_dZ = dC_dZexpanded.reshape(dC_dA.shape[1],-1).T
			
			assert dC_dZ.shape == self.prev_layer.output.shape, f'Back propagating dC_dZ has shape: {dC_dZ.shape} when previous layer output has shape {self.prev_layer.output.shape}'
			if self.FUNCTION is None:
				assert np.array_equal(dC_dZ,dC_dA), 'For activation: None; dC/dZ is expected to be the same as dC/dA.'
			
			return dC_dZ

