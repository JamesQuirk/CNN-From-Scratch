import numpy as np


def array_init(shape: tuple,method=None,seed=None) -> np.ndarray:
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
		raise NameError('ERROR: Unrecognised array initialisation method: ' + method)

	return array

def dilate(array: np.ndarray,channel_width: int) -> np.ndarray:
	""" Inserts 'channel_width' number of 0s between each item in 'array'. """
	_,_, rows, cols = array.shape
	dilation_idx_row = np.arange(rows-1) + 1	# Intiatial indices for insertion of zeros
	dilation_idx_col = np.arange(cols-1) + 1	# Intiatial indices for insertion of zeros
	dilated_array = array.copy()
	for n in range(1,channel_width):	# the n multiplier is to increment the indices in the non-uniform manner required.
		dilated_array = np.insert(
			np.insert( dilated_array, dilation_idx_row * n, 0, axis=2 ),
			dilation_idx_col * n, 0, axis=3)
	return dilated_array
