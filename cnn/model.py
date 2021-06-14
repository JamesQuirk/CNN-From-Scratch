import numpy as np
import pickle
import math
from datetime import datetime as dt
from . import layers
from . import optimisers


# CLASS
class Model():
	"""
	This is the top level class.
	"""

	def __init__(self,optimiser_method='gd'):
		'''
		- optimiser_method (str): Options: ('gd','momentum','rmsprop','adam'). Default is 'gd'.
		'''
		assert optimiser_method.lower() in Model.SUPPORTED_OPTIMISERS, f'You must provide an optimiser that is supported. The options are: {Model.SUPPORTED_OPTIMISERS}'

		self.is_prepared = False

		self.OPTIMISER_METHOD = optimiser_method.lower()

		self.structure = []	# defines order of model (list of layer objects) - EXCLUDES INPUT DATA
		self.layer_counts = dict(zip(['total'] + layers.layers,[0]*(len(layers.layers)+1)))	# dict for counting number of each layer type

	def add_layer(self,layer):
		if layer.LAYER_TYPE == 'ACTIVATION' and self.structure[-1].LAYER_TYPE == 'ACTIVATION':
			print('-- WARNING:: Two Activation Layers in subsequent positions in the model.')
			if layer.FUNCTION == self.structure[-1].FUNCTION:
				print('--- INFO:: Both Activation Layers are the same, skipping creation of second layer.')
				return

		layer.model = self

		if len(self.structure) > 0:
			if layer.__class__.__name__ == 'FC' and self.structure[-1].__class__.__name__ not in ('Flatten','FC','Activation'):
				# If no Flatten layer added before adding first FC layer, one will be added automatically.
				self.add_layer(layers.Flatten())

		self.structure.append(layer)
		self.layer_counts[layer.__class__.__name__] += 1
		self.layer_counts['total'] += 1

		if layer.__class__.__name__ == 'FC':
			# Create the Activation Layer (transparent to user).
			self.add_layer(
				layers.Activation(function=layer.ACTIVATION)
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
		
	def prepare_model(self,optimiser=optimisers.GradientDescent()):
		""" Called once final layer is added, each layer can now initiate its weights and biases. """
		print('Preparing model...')

		assert optimiser.__class__.__name__ in optimisers.optimisers, f'Invalid optimiser: {optimiser}'
		self.OPTIMISER = optimiser

		self.details = {
			'param_counts': [],
			'output_shapes': []
		}
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

				# print(f'Preparing Layer:: Type = {curr_layer.LAYER_TYPE} | Structure index = {curr_layer.MODEL_STRUCTURE_INDEX}')
				curr_layer.prepare_layer()
				# print('--> Num params:',curr_layer.NUM_PARAMS)
				# print('--> Expected output shape:',curr_layer.OUTPUT_SHAPE)
				if curr_layer.MODEL_STRUCTURE_INDEX == 0:
					# First layer; set model input shape.
					self.INPUT_SHAPE = curr_layer.INPUT_SHAPE
				# self.details['param_counts'].append(curr_layer.NUM_PARAMS)
				# self.details['output_shapes'].append(curr_layer.OUTPUT_SHAPE)

		self.OPTIMISER.model = self	# Must be set after all layers are prepared.

		self.is_prepared = True
		# print(self.details)
		self.print_summary()
		print(f'Model Prepared: {self.is_prepared}')

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
		assert self.structure[-1].__class__.__name__ in ('FC','Activation'), 'Model must have either FC or ACTIVATION as final layer.'
		assert Xs.shape[0] == ys.shape[0], f'Dimension (0) of input data [{Xs.shape}] and labels [{ys.shape}] does not match.'
		assert Xs.ndim in (2,4), 'Xs must be either 2 dimensions (for NN) or 4 dimensions (for Model).'
		if Xs.ndim == 4:
			assert Xs.shape[1:] == self.INPUT_SHAPE, f'Expected X shape to be: {self.INPUT_SHAPE}, instead received: {Xs.shape[1:]}'
		elif Xs.ndim == 2:
			assert (Xs.shape[1],1) == self.INPUT_SHAPE, f'Expected X shape to be: {self.INPUT_SHAPE}, instead received: {(Xs.shape[1],1)}'
		assert ys.ndim == 2, f'ys should be a 2 dimensional array.'
		assert type(epochs) == int, 'An integer value must be supplied for argument "epochs"'
		assert int(max_batch_size) == max_batch_size and max_batch_size is not None, 'An integer value must be supplied for argument "max_batch_size"'
		assert cost_fn.lower() in Model.SUPPORTED_COST_FUNCTIONS, f'Chosen cost function not supported, please choose: {Model.SUPPORTED_COST_FUNCTIONS}'
		
		
		# --------- ASSIGNMENTS ----------
		self.N = Xs.shape[0]	# Total number of examples in Xs
		(self.Xs, self.ys) = Model.shuffle(Xs,ys,random_seed) if shuffle else (Xs, ys)
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

	def _print_train_progress(self,batch_index):
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
			self._print_train_progress(batch_ind)

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

	def print_summary(self):
		# layer index | layer type | output shape | param #
		field_names = ['Index','Layer Type','Output Shape','Tr. Param #','Non-Tr. Param #']
		field_lengths = [11,26,21,16,16]	# Includes margin size 1

		print('='*(np.sum(field_lengths) + len(field_names)))
		headers = ''
		for fi in range(len(field_names)):
			headers += ' ' + field_names[fi] + ' '*(field_lengths[fi] - len(field_names[fi]) - 1)	# '-1' to account for leading space.
		print(headers)
		print('='*(np.sum(field_lengths) + len(field_names)))	# +1 for each ' ' that proceeds the field names

		# Add layer info...
		total_trainable = 0
		total_non_trainable = 0
		for layer in self.structure:
			index = str(layer.MODEL_STRUCTURE_INDEX)
			type_ = layer.LAYER_TYPE
			out_shape = layer.OUTPUT_SHAPE
			trainable_params = 0
			non_trainable_params = 0
			for _,param in layer.params.items():
				if param['trainable']:
					trainable_params += param['values'].size
				else:
					non_trainable_params += param['values'].size
			total_trainable += trainable_params
			total_non_trainable += non_trainable_params
			info_str = ' ' + index + ' '*(field_lengths[0] - len(index)-1) + \
				' ' + type_ + ' '*(field_lengths[1] - len(type_) -1) + \
				' ' + str(out_shape) + ' '*(field_lengths[2] - len(str(out_shape))-1) + \
				' ' + str(trainable_params) + ' '*(field_lengths[3] - len(str(trainable_params))-1) + \
				' ' + str(non_trainable_params) + ' '*(field_lengths[4] - len(str(non_trainable_params))-1)
			if index != '0':
				print('-'*(np.sum(field_lengths) + len(field_names)))
			print(info_str)
		print('='*(np.sum(field_lengths) + len(field_names)))
		print('Total params:',total_trainable + total_non_trainable)
		print('Trainable params:',total_trainable)
		print('Non-trainable params:',total_non_trainable)
		print('='*(np.sum(field_lengths) + len(field_names)))