"""
Test case 1:
- 2 filters (2,3,3)
- (2,5,5) input
- stride 2
- No padding
"""


import pytest
import numpy as np
from cnn.layers import Conv2D, conv
from cnn.params import CNNParam


@pytest.fixture
def input_shape():
	return (2,5,5)

@pytest.fixture
def filter_shape():
	return (3,3)

@pytest.fixture
def num_filters():
	return 2

@pytest.fixture
def conv_layer(input_shape,filter_shape,num_filters):
	layer = Conv2D(
			filt_shape=filter_shape,
			num_filters=num_filters,
			stride=2,
			padding=0,
			random_seed=42,
			input_shape=input_shape,
			vectorised=False,
			track_history=False
		)
	layer.prepare_layer()
	return layer

@pytest.fixture
def filters(input_shape,filter_shape,num_filters):
	f = np.zeros((num_filters,input_shape[0],*filter_shape))
	f[0,:,:,:] = 0.5
	f[1,:,:,:] = 0.25
	return f

@pytest.fixture
def bias(num_filters):
	return np.array([[0.2]]*num_filters)

@pytest.fixture
def forwards_input(input_shape):
	""" Representing a selection of input examples """
	n = 2	# Number of examples
	return np.arange(n*np.prod(input_shape)).reshape((n,*input_shape))

@pytest.fixture
def forwards_expected_result():
	return np.array(
			[
				[
					[
						[ 166.7, 184.7],
						[ 256.7, 274.7]
					],
					[
						[ 83.45, 92.45],
						[ 128.45, 137.45]
					]
				],
				[
					[
						[ 616.7, 634.7],
						[ 706.7, 724.7]
					],
					[
						[ 308.45, 317.45],
						[ 353.45, 362.45]
					]
				]
			]
		)

@pytest.fixture
def backwards_input(forwards_expected_result):
	""" Representing example cost gradient of same shape as forwards_expected_result """
	return np.arange(np.prod(forwards_expected_result.shape)).reshape(forwards_expected_result.shape)

@pytest.fixture
def backwards_expected_result():
	return np.array([
		[
			[
				[1., 1., 2.75, 1.75, 1.75],
				[1., 1., 2.75, 1.75, 1.75],
				[3.5, 3.5, 8.5, 5., 5.],
				[2.5, 2.5, 5.75, 3.25, 3.25],
				[2.5, 2.5, 5.75, 3.25, 3.25]
			],
			[
				[1., 1., 2.75, 1.75, 1.75],
				[1., 1., 2.75, 1.75, 1.75],
				[3.5, 3.5, 8.5, 5., 5.],
				[2.5, 2.5, 5.75, 3.25, 3.25],
				[2.5, 2.5, 5.75, 3.25, 3.25]
			]
		],
		[
			[
				[7., 7., 14.75, 7.75, 7.75],
				[7., 7., 14.75, 7.75, 7.75],
				[15.5, 15.5, 32.5, 17., 17.],
				[8.5, 8.5, 17.75, 9.25, 9.25],
				[8.5, 8.5, 17.75, 9.25, 9.25]
			],
			[
				[7., 7., 14.75, 7.75, 7.75],
 				[7., 7., 14.75, 7.75, 7.75],
				[15.5, 15.5, 32.5, 17., 17.],
				[8.5, 8.5, 17.75, 9.25, 9.25],
				[8.5, 8.5, 17.75, 9.25, 9.25],
			]
		]
	])

@pytest.fixture
def backwards_expected_filters_gradient():
	return np.array([
		[
			[
				[2208, 2252, 2296],
				[2428, 2472, 2516],
				[2648, 2692, 2736]
			],
			[
				[3308, 3352, 3396],
				[3528, 3572, 3616],
				[3748, 3792, 3836]
			]
		],
		[
			[
				[3200, 3276, 3352],
				[3580, 3656, 3732],
				[3960, 4036, 4112]
			],
			[
				[5100, 5176, 5252],
				[5480, 5556, 5632],
				[5860, 5936, 6012]
			]
		]
	])

@pytest.fixture
def backwards_expected_bias_gradient():
	return np.array([
		[44],
		[76]
	])

def test_param_class_persistance(conv_layer):
	assert isinstance(conv_layer.filters,CNNParam)
	assert isinstance(conv_layer.bias,CNNParam)
	conv_layer.filters = [1,2,3]
	assert isinstance(conv_layer.filters,CNNParam)
	conv_layer.bias = [1,2,3]
	assert isinstance(conv_layer.bias,CNNParam)

def test_forwards(conv_layer, forwards_input, filters, bias, forwards_expected_result):
	# Set the filters to make them conveient for hand calculations
	conv_layer.filters = filters
	# Set bias to be 0.2 for both
	conv_layer.bias = bias

	assert np.array_equal(
		conv_layer._forwards(forwards_input),
		forwards_expected_result
	)

def test_vec_forwards(conv_layer, forwards_input, filters, bias, forwards_expected_result):
	# Set layer to use vectorised function
	conv_layer.VECTORISED = True
	# Set the filters to make them conveient for hand calculations
	conv_layer.filters = filters
	# Set bias to be 0.2 for both
	conv_layer.bias = bias

	assert np.array_equal(
		conv_layer._forwards(forwards_input),
		forwards_expected_result
	)

def test_backwards(conv_layer, backwards_input, forwards_input, forwards_expected_result, filters, bias, backwards_expected_result, backwards_expected_filters_gradient, backwards_expected_bias_gradient):
	# Setup
	conv_layer.input = forwards_input
	conv_layer.padded_input = forwards_input
	conv_layer.filters = filters
	conv_layer.filters.trainable = False
	conv_layer.bias = bias
	conv_layer.bias.trainable = False
	conv_layer.output = forwards_expected_result

	# Ensure correct output (dCdX)
	assert np.array_equal(
		conv_layer._backwards(backwards_input),
		backwards_expected_result
	)
	# Ensure correct internal cost gradients
	assert np.array_equal(
		conv_layer.filters.gradient,
		backwards_expected_filters_gradient
	)
	assert np.array_equal(
		conv_layer.bias.gradient,
		backwards_expected_bias_gradient
	)

def test_vec_backwards(conv_layer, backwards_input, forwards_input, forwards_expected_result, filters, bias, backwards_expected_result, backwards_expected_filters_gradient, backwards_expected_bias_gradient):
	# Setup
	conv_layer.input = forwards_input
	conv_layer.padded_input = forwards_input
	conv_layer.filters = filters
	conv_layer.filters.trainable = False
	conv_layer.bias = bias
	conv_layer.bias.trainable = False
	conv_layer.VECTORISED = True
	conv_layer.output = forwards_expected_result

	# Ensure correct output (dCdX)
	assert np.array_equal(
		conv_layer._backwards(backwards_input),
		backwards_expected_result
	)
	# Ensure correct internal cost gradients
	assert np.array_equal(
		conv_layer.filters.gradient,
		backwards_expected_filters_gradient
	)
	assert np.array_equal(
		conv_layer.bias.gradient,
		backwards_expected_bias_gradient
	)
