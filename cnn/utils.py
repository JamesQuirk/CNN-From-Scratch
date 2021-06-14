import numpy as np
import pickle

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

def shuffle(X,y,random_seed=None):
	if random_seed is not None:
		np.random.seed(random_seed)
	permutation = np.random.permutation( X.shape[0] )
	X_shuffled = X[permutation]
	y_shuffled = y[permutation]
	print(X_shuffled.shape,y_shuffled.shape)
	assert X.shape == X_shuffled.shape, f'X shape: {X.shape} | X shuffled shape: {X_shuffled.shape}'
	return (X_shuffled, y_shuffled)


def array_init(shape,method=None,seed=None):
	''' Random initialisation of weights array.
	Xavier or Kaiming: (https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79) '''
	assert len(shape) >= 2
	fan_in = shape[-1]
	fan_out = shape[-2]

	if seed:
		np.random.seed(seed)

	if method is None:
		array = np.random.randn(*shape) * 0.01
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

	# print(f'--> Array init method: {method}, max: {array.max()}, min: {array.min()}, std: {array.std()}' )
	# print('Array:',array)
	return array

def load_model(name):
	assert name.split('.')[-1] == 'pkl'
	with open(name, 'rb') as file:  
		model = pickle.load(file)
	return model