import math
import numpy as np
from typing import AnyStr, Tuple

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
	for n in range(1,channel_width+1):	# the n multiplier is to increment the indices in the non-uniform manner required.
		dilated_array = np.insert(
			np.insert( dilated_array, dilation_idx_row * n, 0, axis=2 ),
			dilation_idx_col * n, 0, axis=3)
	return dilated_array

def determine_padding(pad_type: AnyStr,pad_size: int,shape_array_1: Tuple[int],shape_array_2: Tuple[int],stride: int) -> Tuple[int]:
	""" Function to determine required padding at each edge of the array, according to the specified requirements. 
	array_1 refers to the larger of the two arrays that will have array_2 slide over it. """
	if pad_type is None:
		col_left_pad = col_right_pad = row_up_pad = row_down_pad = pad_size
	else:
		if pad_type == 'same':
			nopad_out_cols = math.ceil(float(shape_array_1[1]) / float(stride))
			pad_cols_needed = max((nopad_out_cols - 1) * stride + shape_array_2[1] - shape_array_1[1], 0)
			nopad_out_rows = math.ceil(float(shape_array_1[0]) / float(stride))
			pad_rows_needed = max((nopad_out_rows - 1) * stride + shape_array_2[0] - shape_array_1[0], 0)
		elif pad_type == 'valid':
			# TensoFlow definition of this is "no padding". The input is just processed as-is.
			pad_rows_needed = pad_cols_needed = 0
		elif pad_type == 'include':
			# Here we will implement the padding method to avoid input data being excluded/ missed by the convolution.
			# - This happens when, (I_dim - F_dim) % stride != 0
			if (shape_array_1[0] - shape_array_2[0]) % stride != 0:
				pad_rows_needed = shape_array_2[0] - ((shape_array_1[0] - shape_array_2[0]) % stride)
			else:
				pad_rows_needed = 0
			if (shape_array_1[1] - shape_array_2[1]) % stride != 0:
				pad_cols_needed = shape_array_2[1] - ((shape_array_1[1] - shape_array_2[1]) % stride)
			else:
				pad_cols_needed = 0

		col_left_pad = pad_cols_needed // 2	# // Floor division
		col_right_pad = math.ceil(pad_cols_needed / 2)
		row_up_pad = pad_rows_needed // 2	# // Floor division
		row_down_pad = math.ceil(pad_rows_needed / 2)
	return col_left_pad, col_right_pad, row_up_pad, row_down_pad
