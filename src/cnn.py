'''
This is the main class file for the Convolutional Neural Network

CNN Flow:
	Input -> Conv. -> Pooling [-> Conv. -> Pooling] -> Flatten -> Fully Connected layer -> Output

Array indexing convention: (rows,columns) <- consistent with numpy.
'''

# IMPORTS
import numpy as np
import math

# CLASS
class CNN():
	"""
	This is the top level class. It contains sub-classes for each of the layers that are to be included in the model.
	"""

	def __init__(self,input_shape: tuple,learning_rate=0.01,cost_fn='mse'):
		assert len(input_shape) == 3, 'input_shape must be of length 3: (num_channels, num_rows, num_columns)'

		self.prepared = False

		self.INPUT_SHAPE = input_shape	# tuple to contain input shape
		self.LEARNING_RATE = learning_rate
		self.cost_fn = cost_fn

		self.structure = []	# defines order of model (list of layer objects) - EXCLUDES INPUT DATA
		self.num_layers = {'total':0,'Conv':0,'Pool':0,'Flat':0,'FC':0}	# dict for counting number of each layer type

		# self.overall_cost = None 	# Overall cost of whole model.
		# self.example_cost = None	# Cost of individual example.

	def add_layer(self,layer):
		layer.model = self

		if layer.LAYER_TYPE == 'FC' and self.structure[-1].LAYER_TYPE in ('Conv','Pool'):
			# If no Flatten layer added before adding FC layer, one will be added automatically.
			self.add_layer(CNN.Flatten_Layer())

		self.structure.append(layer)
		self.num_layers[layer.LAYER_TYPE] += 1
		self.num_layers['total'] += 1

		layer.model_structure_index = len(self.structure)-1
		
	def prepare_model(self):
		""" Called once final layer is added, each layer can now iniate its weights and biases. """
		print('Preparing model...')
		if self.num_layers['total'] > 1:
			for index in range(self.num_layers['total'] - 1):
				curr_layer = self.structure[index]
				next_layer = self.structure[index + 1]

				curr_layer.next_layer = next_layer
				next_layer.prev_layer = curr_layer
		
		for layer in self.structure:	# Combine with loop above??
			print(f'Preparing Layer:: Type = {layer.LAYER_TYPE} | Structure index = {layer.model_structure_index}')
			layer.prepare_layer()

		self.prepared = True
		print(f'Model Prepared: {self.prepared}')

	@staticmethod
	def activation(arr,activation,derivative=False):
		# Pass arr through activation function
		if activation == 'relu':
			if not derivative:
				arr[arr<0] = 0
				return arr
			else:
				arr[arr <= 0] = 0	# technically diff. relu is undefined for x = 0, but for the sake of this we use 0 (chances of the weights being exactly 0 is very unlikely anyway).
				arr[arr > 0] = 1
				return arr
		elif activation == 'softmax':
			if not derivative:
				exp = np.exp(arr)
				return exp / np.sum(exp)
			else:
				# TODO: Define derivative
				pass
		elif activation == 'sigmoid':
			if not derivative:
				return 1 / (1 + np.exp(-arr))
			else:
				# TODO: Define derivative
				pass

	def train(self,Xs,ys,epochs,batch_size=None,shuffle=False):
		'''
		Should take array of inputs and array of labels of the same length.

		[For each epoch] For each input, propogate forwards and backwards.

		ARGS:
		- Xs: (N,ch,row,cols). Where N is num examples, ch is num channels.
		- ys: (N,num_categories). e.g. ex1 label = [0,0,0,1,0] for 5 categories.
		'''
		if not self.prepared:
			self.prepare_model()

		# Check shapes and orientation are as expected
		assert Xs.shape[0] == ys.shape[0], 'Dimension of input data and labels does not match.'

		N = Xs.shape[0]	# Total number of examples in Xs

		if batch_size is None:
			num_batches = 1
			self.batch_size = N
		else:
			assert int(batch_size) == batch_size, 'An integer value must be supplied for argument "batch_size"'
			self.batch_size = batch_size

		# Forwards pass...
		for epoch_ind in range(epochs):
			print(f'------ EPOCH: {epoch_ind + 1} ------')
			for batch_ind in range(num_batches):
				ind_lower = batch_ind * self.batch_size	# Lower bound of index range
				ind_upper = np.min([ batch_ind * self.batch_size + self.batch_size , N-1 ])	# Upper bound of index range

				batch_Xs = Xs[ ind_lower : ind_upper ]
				batch_ys = ys[ ind_lower : ind_upper ]

				cost = 0
				cost_gradient = 0

				for ex_ind , X in enumerate(batch_Xs):	# For each example (observation)
					for layer in self.structure:
						X = layer._forwards(X)

					cost += self.cost(X, batch_ys[ex_ind])
					cost_gradient += self.cost(X, batch_ys[ex_ind],derivative=True)	# partial diff of cost w.r.t. activation output of the layer

				print(f'-- Batch: {batch_ind + 1} | Cost: {cost}')

				# Backpropagate the cost
				dcda = cost_gradient
				for layer in self.structure:
					dcda = layer._backwards(dcda)

	def cost(self,prediction,label,derivative=False):
		'''
		Cost function to provide measure of model 'correctness'. returns scalar cost value.
		'''
		if self.cost_fn == 'mse':
			error = label - prediction
			if not derivative:
				return ( np.square( error ) ).sum() / self.batch_size
			else:
				return ( 2 * error ).sum() / self.batch_size
		

	class Conv_Layer:
		def __init__(self,filt_shape: tuple,num_filters: int=5,stride: int=1,padding: int=0,pad_type: str=None):
			""" Padding is determined by the value of 'padding' argument unless 'pad_type' is specified. """
			self.model = None

			self.LAYER_TYPE = 'Conv'
			self.FILT_SHAPE = filt_shape	# 2D tuple describing num rows and cols
			self.NUM_FILTERS = num_filters
			self.STRIDE = stride
			self.PADDING = padding
			if pad_type:
				self.PAD_TYPE = pad_type.lower()
			else:
				self.PAD_TYPE = None

			self.next_layer = None
			self.prev_layer = None

			self.output = None

		def prepare_layer(self):
			if self.prev_layer == None:	# This means this is the first layer in the structure, so 'input' is the only thing before.
				ishape = self.model.INPUT_SHAPE		# (channels, rows, cols)
			else:
				ishape = self.prev_layer.output.shape[-3:]		# (channels, rows, cols)

			# Initiate filters
			filts = []
			for _ in range(self.NUM_FILTERS):
				filts.append( np.random.normal(size=(ishape[0],self.FILT_SHAPE[0], self.FILT_SHAPE[1]) ) )
			self.filters = np.array(filts)
			self.bias = np.random.normal(size=(1,self.NUM_FILTERS))

			# Need to account for padding.
			if self.PAD_TYPE != None:
				if self.PAD_TYPE == 'same':
					out_cols = math.ceil(float(ishape[2]) / float(self.STRIDE))
					pad_cols_needed = max((out_cols - 1) * self.STRIDE + self.FILT_SHAPE[1] - ishape[2], 0)
					self.COL_LEFT_PAD = pad_cols_needed // 2	# // Floor division
					self.COL_RIGHT_PAD = math.ceil(pad_cols_needed / 2)

					out_rows = math.ceil(float(ishape[1]) / float(self.STRIDE))
					pad_rows_needed = max((out_rows - 1) * self.STRIDE + self.FILT_SHAPE[0] - ishape[1], 0)
					self.ROW_DOWN_PAD = pad_rows_needed // 2	# // Floor division
					self.ROW_UP_PAD = math.ceil(pad_rows_needed / 2)
				elif self.PAD_TYPE == 'valid':
					pass # TODO: This allows the filter to fit into the input size an integer number of times?? CHECK.
			else:
				self.COL_LEFT_PAD = self.COL_RIGHT_PAD = self.ROW_UP_PAD = self.ROW_DOWN_PAD = self.PADDING

			col_out = int((ishape[2] + (self.COL_LEFT_PAD + self.COL_RIGHT_PAD) - self.FILT_SHAPE[1]) / self.STRIDE) + 1
			row_out = int((ishape[1] + (self.ROW_DOWN_PAD + self.ROW_UP_PAD) - self.FILT_SHAPE[0]) / self.STRIDE) + 1

			self.output = np.zeros(shape=(self.NUM_FILTERS,row_out,col_out))	# Output initiated.
			print(self.output.shape)

		def _forwards(self,_input):
			self._input = _input

			# Apply the padding to the input.
			if _input.ndim == 3:
				self.padded_input = np.pad(_input,[(0,0),(self.ROW_UP_PAD,self.ROW_DOWN_PAD),(self.COL_LEFT_PAD,self.COL_RIGHT_PAD)],'constant')
			elif _input.ndim == 2:
				self.padded_input = np.pad(_input,[(self.ROW_UP_PAD,self.ROW_DOWN_PAD),(self.COL_LEFT_PAD,self.COL_RIGHT_PAD)],'constant')
			else:
				print('---ERROR:: _input array does not have a suitable number of dimensions.')

			_, proc_rows, proc_cols = self.padded_input.shape

			for filt_index in range(self.NUM_FILTERS):
				filt = self.filters[filt_index]
				filt_channels, filt_rows, filt_cols = filt.shape

				for channel_index in range(filt_channels):
					self.output[filt_index] += Conv_Layer.convolve( self.padded_input[channel_index], filt[channel_index], self.STRIDE )
				
				self.output[filt_index] += self.bias[filt_index]

				return self.output	# Output is 3D array of shape: ( NUM_FILTS, NUM_ROWS, NUM_COLS )

		def _backwards(self,cost_gradient):
			dCdF = []	# initiate as list then convert to np.array
			for channel_index in range(self.padded_input.size[0]):
				dCdF.append( Conv_Layer.convolve( self.padded_input[channel_index], cost_gradient[channel_index] ) )
			dCdF = np.array( dCdF )

			self.filters = self.filters + ( self.model.LEARNING_RATE * dCdF	) # ADJUSTING THE FILTERS
			
			# Find cost gradient wrt previous output.
			rot_F = np.rot90( self.filters, k=2, axes=(1,2) )	# rotate 2x90 degs, rotating in direction of rows to columns.
			dCdX = Conv_Layer.convolve( rot_F, cost_gradient, full_convolve=True )
			
			return dCdX

		@staticmethod
		def convolve(A, B, stride,full_convolve=False):
			""" A and B are 2D arrays. Array B will be convolved over Array A using the stride provided.
				- 'full_convolve' is where the bottom right cell of B starts over the top of the top left cell of A and shifts by stride until the top left cell of B is over the bottom right cell of A. (i.e. A is padded in each dimension by B - 1 in the respective dimension). """
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
			self.model = None

			self.LAYER_TYPE = 'Pool'
			self.FILT_SHAPE = filt_shape	# 2D array (rows,cols)
			self.STRIDE = stride
			self.POOL_TYPE = pool_type.lower()
			self.PADDING = padding
			if pad_type:
				self.PAD_TYPE = pad_type.lower()
			else:
				self.PAD_TYPE = None

			self.next_layer = None
			self.prev_layer = None

		def prepare_layer(self):
			""" This needs to be done after the input has been identified - currently happens when train() is called. """
			if self.prev_layer == None:	# This means this is the first layer in the structure, so 'input' is the only thing before.
				ishape = self.model.INPUT_SHAPE		# (channels, rows, cols)
			else:
				ishape = self.prev_layer.output.shape		# (channels, rows, cols)

			# Need to account for padding.
			if self.PAD_TYPE != None:
				if self.PAD_TYPE == 'same':
					out_cols = math.ceil(float(ishape[2]) / float(self.STRIDE))
					pad_cols_needed = max((out_cols - 1) * self.STRIDE + self.FILT_SHAPE[1] - ishape[2], 0)
					self.COL_LEFT_PAD = pad_cols_needed // 2	# // Floor division
					self.COL_RIGHT_PAD = math.ceil(pad_cols_needed / 2)

					out_rows = math.ceil(float(ishape[1]) / float(self.STRIDE))
					pad_rows_needed = max((out_rows - 1) * self.STRIDE + self.FILT_SHAPE[0] - ishape[1], 0)
					self.ROW_DOWN_PAD = pad_rows_needed // 2	# // Floor division
					self.ROW_UP_PAD = math.ceil(pad_rows_needed / 2)
			else:
				self.COL_LEFT_PAD = self.COL_RIGHT_PAD = self.ROW_UP_PAD = self.ROW_DOWN_PAD = self.PADDING

			col_out = int((ishape[2] + (self.COL_LEFT_PAD + self.COL_RIGHT_PAD) - self.FILT_SHAPE[1]) / self.STRIDE) + 1
			row_out = int((ishape[1] + (self.ROW_DOWN_PAD + self.ROW_UP_PAD) - self.FILT_SHAPE[0]) / self.STRIDE) + 1

			self.output = np.zeros(shape=(ishape[0],row_out,col_out))	# Output initiated.
			print(self.output.shape)

		def _forwards(self,_input):
			self._input = _input

			# Apply the padding to the input.
			if _input.ndim == 3:
				self.padded_input = np.pad(_input,[(0,0),(self.ROW_UP_PAD,self.ROW_DOWN_PAD),(self.COL_LEFT_PAD,self.COL_RIGHT_PAD)],'constant')
			elif _input.ndim == 2:
				self.padded_input = np.pad(_input,[(self.ROW_UP_PAD,self.ROW_DOWN_PAD),(self.COL_LEFT_PAD,self.COL_RIGHT_PAD)],'constant')
			else:
				print('---ERROR:: _input array does not have a suitable number of dimensions.')

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
			# Initiate to input shape.
			prev_cost_gradient = np.zeros_like(self.padded_input)

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
							max_node_y, max_node_x = np.array( np.unravel_index( sub_arr, sub_arr.shape ) ) + np.array([curr_y, curr_x])	# addition of curr_y & curr_x is to get position in padded_input array (not just local sub_arr).

							cost_val = cost_gradient[channel_index,cost_y,cost_x]

							prev_cost_gradient[channel_index, max_node_y, max_node_x] += cost_val
							
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
			self.LAYER_TYPE = 'Flat'

			self.next_layer = None
			self.prev_layer = None

		def prepare_layer(self):
			self.output = np.zeros(shape=(self.prev_layer.output.size,1))
			pass 

		def _forwards(self,_input):
			return _input.reshape((_input.size,1))	# NOTE: Vertical array

		def _backwards(self,cost_gradient):
			return cost_gradient.reshape(self.prepare_layer.output.shape)	# TODO: Test


	class FC_Layer:
		"""
		The Fully Connected Layer is defined as being the layer of nodes and the weights of the connections that link those nodes to the previous layer.
		"""
		def __init__(self, n, activation='relu'):
			self.model = None

			self.LAYER_TYPE = 'FC'
			self.NUM_NODES = n
			self.ACTIVATION = activation.lower()

			self.next_layer = None
			self.prev_layer = None

			self.weights = None
			self.bias = None

			self.output = np.zeros(shape=(n,1))

		def prepare_layer(self):
			""" Initiate weights and biases randomly"""
			w_cols = self.prev_layer.output.size
			w_rows = self.NUM_NODES	# "Each row corresponds to all connections of previous layer to a single node in current layer."
			self.weights = np.random.normal(size=(w_rows,w_cols))

			self.bias = np.random.normal(size=(1,self.NUM_NODES))
			print(self.output.shape)

		def _forwards(self,_input):
			self._input = _input

			self.output = CNN.activation( np.dot( self.weights, self._input ) + self.bias , activation=self.ACTIVATION )

			return self.output

		def _backwards(self, cost_gradient):
			"""
			Take cost gradient dC/da (how the activation affects the cost) and backpropogate
			"""
			dzdw = self.prev_layer.output	# Activation output of previous layer
			z = np.multiply( self.weights, self.prev_layer.output ) + self.bias
			dadz = CNN.activation( z, activation=self.ACTIVATION, derivative=True )
			dcda = cost_gradient
			
			dcdw = dzdw * dadz * dcda	# Chain rule.

			self.weights = self.weights - ( self.model.LEARNING_RATE * dcdw )
			
			# calculate cost_gradient wrt previous layer
			dzda_prev = self.weights	# TODO: CHECK THIS?? - does this depend on cost function?
			prev_cost_gradient = dzda_prev * dadz * dcda

			return prev_cost_gradient	# NOTE: Returns a 1D cost gradient array


