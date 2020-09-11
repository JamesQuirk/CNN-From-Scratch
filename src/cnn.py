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
	---- TEST ----
	>>> model = CNN()
	>>> model.add_layer( CNN.Conv_Layer( (2,2), 1, 1 ) )
	>>> model.add_layer( CNN.Pool_Layer( (2,2), 1 ) )
	>>> model.add_layer( CNN.FC_Layer( 4 ) )
	>>> model.prepare_model()
	>>> in = np.arange(25).reshape((5,5))
	>>> model.structure[0]._forwards(in)
	"""

	# def __init__(self,n_conv,n_masks,mask_size,conv_stride,ds_size,ds_stride,fc_shape):
	# 	'''
	# 	Params:
	# 		- n_conv: {int}; number of convolution layers (and downsampling layers)
	# 		- n_masks: {int/tuple}; number of masks for convolution.
	# 		- mask_size: {int/tuple}; size of masks for convolution.
	# 		- conv_stride: {int/tuple}; stride length for convolution layers.
	# 		- ds_size: {int/tuple}; size of filter window for downsampling layers.
	# 		- ds_stride: {int/tuple}; stride length for downsampling layers.
	# 		- fc_shape: {tuple}; indicating number of nodes for each layer of fully connected network, excluding input layer. 

	# 	NOTE:: For {int/tuple} parameters: 'int' -> same value for each layer.
	# 		'tuple' -> each value of the tuple is the specific value for that layer respectively.
	# 	'''
	# 	#################################################
	# 	######## Standardise the parameters #############
	# 	#################################################
	# 	if type(n_masks) == int:
	# 		self.n_masks = tuple([n_masks for x in range(n_conv)])
	# 	elif type(n_masks) == tuple:
	# 		if len(n_masks) != n_conv:
	# 			print('ERROR:: length of n_masks tuple does not match n_conv')
	# 		else:
	# 			self.n_masks = n_masks
	# 	else:
	# 		print('ERROR:: Incorrect type provided for n_masks. Expected \'int\' or \'tuple\'.')

	# 	if type(mask_size) == int:
	# 		self.mask_size = tuple([mask_size for x in range(n_conv)])
	# 	elif type(mask_size) == tuple:
	# 		if len(mask_size) != n_conv:
	# 			print('ERROR:: length of mask_size tuple does not match n_conv')
	# 		else:
	# 			self.mask_size = mask_size
	# 	else:
	# 		print('ERROR:: Incorrect type provided for mask_size. Expected \'int\' or \'tuple\'.')

	# 	if type(conv_stride) == int:
	# 		self.conv_stride = tuple([conv_stride for x in range(n_conv)])
	# 	elif type(conv_stride) == tuple:
	# 		if len(conv_stride) != n_conv:
	# 			print('ERROR:: length of conv_stride tuple does not match n_conv')
	# 		else:
	# 			self.conv_stride = conv_stride
	# 	else:
	# 		print('ERROR:: Incorrect type provided for conv_stride. Expected \'int\' or \'tuple\'.')

	# 	if type(ds_size) == int:
	# 		self.ds_size = tuple([ds_size for x in range(n_conv)])
	# 	elif type(ds_size) == tuple:
	# 		if len(ds_size) != n_conv:
	# 			print('ERROR:: length of ds_size tuple does not match n_conv')
	# 		else:
	# 			self.ds_size = ds_size
	# 	else:
	# 		print('ERROR:: Incorrect type provided for ds_size. Expected \'int\' or \'tuple\'.')

	# 	if type(ds_stride) == int:
	# 		self.ds_stride = tuple([ds_stride for x in range(n_conv)])
	# 	elif type(ds_stride) == tuple:
	# 		if len(ds_stride) != n_conv:
	# 			print('ERROR:: length of ds_stride tuple does not match n_conv')
	# 		else:
	# 			self.ds_stride = ds_stride
	# 	else:
	# 		print('ERROR:: Incorrect type provided for ds_stride. Expected \'int\' or \'tuple\'.')

	# 	if type(fc_shape) == tuple:
	# 		if len(fc_shape) < 1:
	# 			print('ERROR:: insufficient values provided for fc_shape')
	# 		else:
	# 			self.fc_shape = fc_shape
	# 	else:
	# 		print('ERROR:: Incorrect type provided for fc_shape. Expected \'tuple\'.')

	# 	## Fully Connected layer
	# 	self.fc_weights = []	# This will be a list of np.array objects

	def __init__(self,input_shape: tuple,learning_rate=0.01,cost_fn='mse'):
		self.prepared = False

		assert len(input_shape) == 3, 'input_shape must be of length 3: (num_channels, num_rows, num_columns)'
		self.INPUT_SHAPE = input_shape	# tuple to contain input shape
		self._input = None
		self.LEARNING_RATE = learning_rate
		self.cost_fn = cost_fn	# Cost function is somewhat arbitrary

		self.structure = []	# defines order of model (list of layer objects) - EXCLUDES INPUT DATA
		self.num_layers = {'total':0,'Conv':0,'Pool':0,'Flat':0,'FC':0}	# dict for counting number of each layer type

		self.cost = None 	# Overall cost of whole model.

	def add_layer(self,layer):
		layer.model = self

		if layer.LAYER_TYPE == 'FC' and self.structure[-1].LAYER_TYPE in ('Conv','Pool'):
			self.add_layer(CNN.Flatten_Layer())

		self.structure.append(layer)
		self.num_layers[layer.LAYER_TYPE] += 1
		self.num_layers['total'] += 1

		if self.num_layers['total'] > 1:
			for index in range(self.num_layers['total'] - 1):
				curr_layer = self.structure[index]
				next_layer = self.structure[index + 1]

				curr_layer.next_layer = next_layer
				next_layer.prev_layer = curr_layer

		
	def prepare_model(self):
		""" Called once final layer is added, each layer can now iniate its weights and biases. """
		for layer in self.structure:
			layer.prepare_layer()

		self.prepared = True


	def train(self,_input,_labels):
		self._input = _input	# this is the input data (usually an 'image') that is passed to the CNN.
		self._labels = _labels

		# Forwards pass...
		x = _input
		for layer in self.structure:
			x = layer._forwards(x)

		# TODO: Calculate cost and backpropogate


	@staticmethod
	def activation(arr,activation,derivative=False):
		# Pass arr through activation function
		if activation == 'relu':
			if not derivative:
				arr[arr<0] = 0
				return arr
			else:
				arr[arr <= 0] = 0	# technically diff. relu is undefined for x = 0, but for the sake of this we use 0 (chances of the weights being exactly 0 is very unlikely anyway.)
				arr[arr > 0] = 1
				return arr
		elif activation == 'softmax':
			if not derivative:
				out = np.exp(arr)
				return out / np.sum(out)
			else:
				pass
		elif activation == 'sigmoid':
			if not derivative:
				return 1 / (1 + np.exp(-arr))
			else:
				pass


	@staticmethod
	def loss(predictions,labels,loss_function):
		'''
		Error of final output = y_label - y_pred
		'''
		pass


	class Conv_Layer:
		def __init__(self,filt_size: tuple,num_filters: int,stride: int,padding: int=0,pad_type: str=None):
			""" Padding is determined by the value of 'padding' argument unless 'pad_type' is specified. """
			self.model = None

			self.LAYER_TYPE = 'Conv'

			self.FILT_SIZE = filt_size	# TODO: Need to ensure the filters have the same number of channels as the input image.
			self.NUM_FILTERS = num_filters
			self.STRIDE = stride
			self.PADDING = padding
			self.PAD_TYPE = pad_type.lower()

			filts = []
			for _ in range(self.NUM_FILTERS):
				filts.append( np.random.normal(size=self.FILT_SIZE) )
			self.filters = np.array(filts)
			self.bias = np.random.normal(size=(1,self.NUM_FILTERS))

			self.next_layer = None
			self.prev_layer = None

			self.output = None


		def prepare_layer(self):
			if self.prev_layer == None:	# This means this is the first layer in the structure, so 'input' is the only thing before.
				ishape = self.model.INPUT_SHAPE		# (channels, rows, cols)
			else:
				ishape = self.prev_layer.output.shape[-3:]		# (channels, rows, cols)

			# Need to account for padding.
			if self.PAD_TYPE != None:
				if self.PAD_TYPE == 'same':
					out_cols = math.ceil(float(ishape[2]) / float(self.STRIDE))
					pad_cols_needed = max((out_cols - 1) * self.STRIDE + self.FILT_SIZE[1] - ishape[2], 0)
					self.COL_LEFT_PAD = pad_cols_needed // 2	# // Floor division
					self.COL_RIGHT_PAD = math.ceil(pad_cols_needed / 2)

					out_rows = math.ceil(float(ishape[1]) / float(self.STRIDE))
					pad_rows_needed = max((out_rows - 1) * self.STRIDE + self.FILT_SIZE[0] - ishape[1], 0)
					self.ROW_DOWN_PAD = pad_rows_needed // 2	# // Floor division
					self.ROW_UP_PAD = math.ceil(pad_rows_needed / 2)
				elif self.PAD_TYPE == 'valid':
					pass # TODO
			else:
				self.COL_LEFT_PAD = self.COL_RIGHT_PAD = self.ROW_UP_PAD = self.ROW_DOWN_PAD = self.PADDING

			col_out = int((ishape[2] + (self.COL_LEFT_PAD + self.COL_RIGHT_PAD) - self.FILT_SIZE[1]) / self.STRIDE) + 1
			row_out = int((ishape[1] + (self.ROW_DOWN_PAD + self.ROW_UP_PAD) - self.FILT_SIZE[0]) / self.STRIDE) + 1

			self.output = np.zeros(size=(self.NUM_FILTERS,row_out,col_out))	# Output initiated.


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

				# # start with mask in top left corner
				# curr_y = out_y = 0	# 'curr_y' is y position of the top left corner of filt on top of '_input'. 'out_y' is the respective y position in the output array.
				# while curr_y <= proc_rows - filt_rows:
				# 	curr_x = out_x = 0	# 'curr_x' is x position of the top left corner of filt on top of '_input'. 'out_x' is the respective x position in the output array.
				# 	while curr_x <= proc_cols - filt_cols:
				# 		for channel_index in range(filt_channels):
				# 			self.output[filt_index,out_y,out_x] += np.sum(self.padded_input[channel_index,curr_y:curr_y+filt_rows,curr_x:curr_x+filt_cols] * filt[channel_index])
				# 		self.output[filt_index,out_y,out_x] += self.bias[filt_index]	# Add the bias
				# 		curr_x += self.STRIDE
				# 		out_x += 1

				# 	curr_y += self.STRIDE
				# 	out_y += 1

				for channel_index in range(filt_channels):
					self.output[filt_index] += CNN.convolve( self.padded_input[channel_index], filt[channel_index], self.STRIDE )
				
				self.output[filt_index] += self.bias[filt_index]

				return self.output	# Output is 3D array of shape: ( NUM_FILTS, NUM_ROWS, NUM_COLS )


		def _backwards(self,cost_gradient):
			if self.next_layer.LAYER_TYPE == 'FC':
				# TODO: Then some kind of reshaping needs to happen. How is the cost gradient passed between FC and Pooling/ Conv Layers??
				# cost_gradient = cost_gradient.reshape(...)
				pass
			
			dCdF = []	# initiate as list then convert to np.array
			for channel_index in range(self.padded_input.size[0]):
				dCdF.append( CNN.convolve( self.padded_input[channel_index], cost_gradient[channel_index] ) )
			dCdF = np.array( dCdF )

			self.filters = self.filters + dCdF	# ADJUSTING THE FILTERS
			
			flipped_F = self.
			dCdX = 


	@staticmethod
	def convolve(A, B, stride):
		""" A and B are 2D arrays. Array B will be convolved over Array A using the stride provided. """
		arows, acols = A.shape
		brows, bcols = B.shape

		rout = int((arows - brows) / stride) + 1
		cout = int((acols - bcols) / stride) + 1

		output = np.zeros(shape=(rout,cout))

		# start with mask in top left corner
		curr_y = out_y = 0	# 'curr_y' is y position of the top left corner of filt on top of '_input'. 'out_y' is the respective y position in the output array.
		while curr_y <= arows - brows:
			curr_x = out_x = 0	# 'curr_x' is x position of the top left corner of filt on top of '_input'. 'out_x' is the respective x position in the output array.
			while curr_x <= acols - bcols:
				output[out_y,out_x] += np.sum( A[ curr_y : curr_y + brows, curr_x : curr_x + bcols ] * B)
				curr_x += stride
				out_x += 1

			curr_y += stride
			out_y += 1

		return output

	
	class Pool_Layer:
		def __init__(self,filt_size: tuple,stride: int,pool_type: str='max',padding: int=0,pad_type: str=None):
			self.model = None

			self.LAYER_TYPE = 'Pool'
			self.FILT_SIZE = filt_size
			self.STRIDE = stride
			self.POOL_TYPE = pool_type.lower()
			self.PADDING = padding
			self.PAD_TYPE = pad_type.lower()

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
					pad_cols_needed = max((out_cols - 1) * self.STRIDE + self.FILT_SIZE[1] - ishape[2], 0)
					self.COL_LEFT_PAD = pad_cols_needed // 2	# // Floor division
					self.COL_RIGHT_PAD = math.ceil(pad_cols_needed / 2)

					out_rows = math.ceil(float(ishape[1]) / float(self.STRIDE))
					pad_rows_needed = max((out_rows - 1) * self.STRIDE + self.FILT_SIZE[0] - ishape[1], 0)
					self.ROW_DOWN_PAD = pad_rows_needed // 2	# // Floor division
					self.ROW_UP_PAD = math.ceil(pad_rows_needed / 2)
			else:
				self.COL_LEFT_PAD = self.COL_RIGHT_PAD = self.ROW_UP_PAD = self.ROW_DOWN_PAD = self.PADDING

			col_out = int((ishape[2] + (self.COL_LEFT_PAD + self.COL_RIGHT_PAD) - self.FILT_SIZE[1]) / self.STRIDE) + 1
			row_out = int((ishape[1] + (self.ROW_DOWN_PAD + self.ROW_UP_PAD) - self.FILT_SIZE[0]) / self.STRIDE) + 1

			self.output = np.zeros(size=(ishape[0],row_out,col_out))	# Output initiated.


		def _forwards(self,_input):
			self._input = _input

			# Apply the padding to the input.
			if _input.ndim == 3:
				padded_input = np.pad(_input,[(0,0),(self.ROW_UP_PAD,self.ROW_DOWN_PAD),(self.COL_LEFT_PAD,self.COL_RIGHT_PAD)],'constant')
			elif _input.ndim == 2:
				padded_input = np.pad(_input,[(self.ROW_UP_PAD,self.ROW_DOWN_PAD),(self.COL_LEFT_PAD,self.COL_RIGHT_PAD)],'constant')
			else:
				print('---ERROR:: _input array does not have a suitable number of dimensions.')

			channels, proc_rows, proc_cols = padded_input.shape

			# Shift 'Filter Window' over the image and perform the downsampling
			curr_y = out_y = 0
			while curr_y <= proc_rows - self.FILT_SIZE[0]:
				curr_x = out_x = 0
				while curr_x <= proc_cols - self.FILT_SIZE[1]:
					for channel_index in range(channels):
						if self.POOL_TYPE == 'max':
							self.output[channel_index, out_y, out_x] = np.max( padded_input[ channel_index, curr_y : curr_y + self.FILT_SIZE[0], curr_x : curr_x+ self.FILT_SIZE[1] ] )
						elif self.POOL_TYPE == 'mean':
							self.output[channel_index, out_y, out_x] = np.mean( padded_input[ channel_index, curr_y : curr_y + self.FILT_SIZE[0], curr_x : curr_x + self.FILT_SIZE[1] ] )

					curr_x += self.STRIDE
					out_x += 1
				curr_y += self.STRIDE
				out_y += 1

			return self.output

		def _backwards(self,cost_gradient):
			pass


	class Flatten_Layer:
		""" A psuedo layer that simply adjusts the data dimension as it passes between 2D/3D Conv or Pool layers to the 1D FC layers. """

		def __init__(self):
			self.LAYER_TYPE = 'Flat'

			self.next_layer = None
			self.prev_layer = None

		def prepare_layer(self):
			pass # No prep needed.

		def _forwards(self,_input):
			return _input.reshape((_input.size,1))

		def _backwards(self,cost_gradient):
			# TODO: should reshape cost_gradient?


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


		def prepare_layer(self):
			""" Initiate weights and biases randomly"""
			w_cols = self.prev_layer.output.size
			w_rows = self.NUM_NODES	# "Each row corresponds to all connections of previous layer to a single node in current layer."
			self.weights = np.random.normal(size=(w_rows,w_cols))

			self.bias = np.random.normal(size=(1,self.NUM_NODES))
			

		def _forwards(self,_input):
			# if self.prev_layer.LAYER_TYPE in ('Conv','Pool'):
			# 	self._input = _input.reshape((_input.size,1))
			# else:
			# 	self._input = _input
			self._input = _input

			self.output = CNN.activation( np.dot( self.weights, self._input ) + self.bias , activation=self.ACTIVATION )

			return self.output

		def _backwards(self, cost_gradient):
			"""
			Take cost gradient dC/da (how the activation affects the cost)
			"""
			dzdw = self.prev_layer.output	# Activation output of previous layer
			z = np.multiply( self.weights, self.prev_layer.output ) + self.bias
			dadz = CNN.activation( z, activation=self.ACTIVATION, derivative=True )
			dcda = cost_gradient
			
			dcdw = dzdw * dadz * dcda	# Chain rule.

			self.weights = self.weights - self.model.LEARNING_RATE * dcdw
			
			# calculate cost_gradient wrt previous layer
			dzda_prev = self.weights
			
			prev_cost_gradient = dzda_prev * dadz * dcda
			return prev_cost_gradient	# Returns a 1D cost gradient array





# -------------------------------- ARCHIVED FUNCTIONS -----------------------------------

	@staticmethod
	def convolution(image,mask_arr,bias,stride=1,padding=0):
		'''
		Convolves each `mask` in 'mask_arr' over `image` by 'stride'
		
		Params
			- image (np.arr): matrix of pixel values of the image.
			- mask_arr (np.arr): array of masks (np.arr): matrix of values
			- bias (np.arr): 1d array of bias values (1 bias for each mask)
			- stride (int): integer value representing the number of pixels to step over on each move.
		'''

		# at this stage, imagine we have the mask already (it will just be passed to this function later on)
		(num_masks, mask_channels, mask_rows, mask_cols) = mask_arr.shape
		(img_channels,img_rows,img_cols) = image.shape

		assert img_channels == mask_channels, "Masks must have the same number of channels as the image."

		# Pad the image
		padded = np.pad(image,[(0,0),(padding,padding),(padding,padding)],'constant')
		(_,pad_rows,pad_cols) = padded.shape

		# Output dimensions are equal for square images and square masks - otherwise use respective dimension in calculation.
		out_rows = int((img_rows + 2*padding - mask_rows)/stride) + 1	# 'int()' acts as 'floor' for the calculation.
		out_cols = int((img_cols + 2*padding - mask_cols)/stride) + 1	# 'int()' acts as 'floor' for the calculation.

		output = np.zeros((num_masks,out_rows,out_cols))

		# Convolve each filter over the image (keep track of top left 'element')
		for mask_index in range(num_masks):
			mask = mask_arr[mask_index]

			# start with mask in top left corner
			curr_y = out_y = 0	
			while curr_y <= pad_rows - mask_rows:
				curr_x = out_x = 0
				while curr_x <= pad_cols - mask_cols:
					for channel_index in range(mask_channels):
						output[mask_index,out_y,out_x] += np.sum(padded[channel_index,curr_y:curr_y+mask_rows,curr_x:curr_x+mask_cols] * mask[channel_index])
					output[mask_index,out_y,out_x] += bias[mask_index]	# Add the bias
					curr_x += stride
					out_x += 1

				curr_y += stride
				out_y += 1

		return output	# a 3D array of the output matrices [1 for each mask].

	@staticmethod
	def downsample(image,windowSize=2,stride=2,downsampling='maxpooling',padding='same'):
		if type(windowSize) == int:
			windowSize = (windowSize,windowSize)
		elif type(windowSize) == tuple:
			if len(windowSize) > 2:
				print('-- WARNING:: tuple provided is too long. Only first 2 elements will be used.')
				windowSize = windowSize[:2]
		else:
			raise TypeError('WindowSize argument has invalid type: %s was provided where int or tuple is expected.' % type(windowSize))
		
		(img_channels, img_rows, img_cols) = image.shape

		# Apply padding in 'same' mode to avoid dropping pixels in the case of odd dimensions. (favour below and right - same as Tensorflow)
		if padding.lower() == 'same':
			out_height = math.ceil(float(img_rows) / float(stride))
			pad_rows_needed = max((out_height - 1) * stride + windowSize[0] - img_rows, 0)
			nbelow = pad_rows_needed // 2	# // Floor division
			nabove = math.ceil(pad_rows_needed / 2)

			out_width = math.ceil(float(img_cols) / float(stride))
			pad_cols_needed = max((out_width - 1) * stride + windowSize[1] - img_cols, 0)
			nleft = pad_cols_needed // 2
			nright = math.ceil(pad_cols_needed / 2)

			padded = np.pad(image,[(0,0),(nabove,nbelow),(nleft,nright)],'constant')
		else:
			padded = image

		(_,pad_rows,pad_cols) = padded.shape

		h_out = int((pad_rows - windowSize[0])/stride) + 1
		w_out = int((pad_cols - windowSize[1])/stride) + 1

		downsampled = np.zeros((img_channels,h_out,w_out))

		# Shift 'Filter Window' over the image and perform the downsampling
		curr_y = out_y = 0
		while curr_y <= img_rows - windowSize[0]:
			curr_x = out_x = 0
			while curr_x <= img_cols - windowSize[1]:
				for channel_index in range(img_channels):
					if downsampling == 'maxpooling':
						downsampled[channel_index,out_y,out_x] = np.max(image[channel_index,curr_y:curr_y+windowSize[0],curr_x:curr_x+windowSize[1]])
					elif downsampling == 'meanpooling':
						downsampled[channel_index,out_y,out_x] = np.mean(image[channel_index,curr_y:curr_y+windowSize[0],curr_x:curr_x+windowSize[1]])

				curr_x += stride
				out_x += 1
			curr_y += stride
			out_y += 1

		return downsampled


	@staticmethod
	def fully_connected_layer(input_arr,weights_arr,bias_arr):
		'''
		Params:
			- input_arr: 1D array. 
			- weights_arr: list of np.array(). Length = (No. layers - 1); No. rows = No. nodes in NEXT layer (b); No. cols = No. nodes in CURRENT layer (a).
			- bias_arr: array of bias. No. bias = len(weights_arr)
		'''

		# Check that input_array is a column vector
		if len(input_arr.shape) != 1:
			(r,c) = input_arr.shape
			assert len(input_arr.shape) > 2, "A matrix of incorrect shape was passed for 'input_arr'"
			assert r+c == max(r,c) + 1, "input_arr must be a 1D array."
		input_arr = input_arr.reshape(np.max(input_arr.shape),1)

		# Proceed through network...
		values_a = input_arr
		for layer_index in range(len(weights_arr)):	# Excluded output layer
			weights = weights_arr[layer_index]	# np.array of weights between layer a and layer b

			values_b = np.dot(weights,values_a)

			# Add bias and pass through activiation. Initiate next layer with result
			if layer_index != len(weights_arr) - 1:
				values_a = CNN.activation( values_b + bias_arr[layer_index] , activation='relu')
			else:
				values_a = CNN.activation( values_b + bias_arr[layer_index] , activation='softmax')

		return values_a
