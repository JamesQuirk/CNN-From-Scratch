import numpy as np
from typing import Tuple

def one_hot_encode(array: np.ndarray,num_cats: int,axis: bool=None) -> np.ndarray:
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

def shuffle(X: np.ndarray,y: np.ndarray,random_seed: bool=None) -> Tuple[np.ndarray]:
	if random_seed is not None:
		np.random.seed(random_seed)
	permutation = np.random.permutation( X.shape[0] )
	X_shuffled = X[permutation]
	y_shuffled = y[permutation]
	print(X_shuffled.shape,y_shuffled.shape)
	assert X.shape == X_shuffled.shape, f'X shape: {X.shape} | X shuffled shape: {X_shuffled.shape}'
	return (X_shuffled, y_shuffled)

