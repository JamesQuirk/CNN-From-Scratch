"""
Test case 2:
- 2 filters (2,2,2)	- (these filters would not fit a whole number of times (with stride 2) into input therefore resulting in data being excluded)
- (2,5,5) input
- stride 2
- Padding 0
"""

from _pytest.recwarn import recwarn
import pytest
import numpy as np
from cnn.layers import Conv2D
from cnn.params import CNNParam


@pytest.fixture
def input_shape():
	return (2,5,5)

@pytest.fixture
def filter_shape():
	return (2,2)

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
					[62.2, 70.2],
					[102.2, 110.2]],
				[
					[31.2, 35.2],
					[51.2, 55.2]
				]
			],
			[
				[
					[262.2, 270.2],
					[302.2, 310.2]],
				[
					[131.2, 135.2],
					[151.2, 155.2]
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
				[1, 1, 1.75, 1.75, 0],
				[1, 1, 1.75, 1.75, 0],
				[2.5, 2.5, 3.25, 3.25, 0],
				[2.5, 2.5, 3.25, 3.25, 0],
				[0, 0, 0, 0, 0]
			],
			[
				[1, 1, 1.75, 1.75, 0],
				[1, 1, 1.75, 1.75, 0],
				[2.5, 2.5, 3.25, 3.25, 0],
				[2.5, 2.5, 3.25, 3.25, 0],
				[0, 0, 0, 0, 0]
			]
		],
		[
			[
				[7, 7, 7.75, 7.75, 0],
				[7, 7, 7.75, 7.75, 0],
				[8.5, 8.5, 9.25, 9.25, 0],
				[8.5, 8.5, 9.25, 9.25, 0],
				[0, 0, 0, 0, 0]
			],
			[
				[7, 7, 7.75, 7.75, 0],
				[7, 7, 7.75, 7.75, 0],
				[8.5, 8.5, 9.25, 9.25, 0],
				[8.5, 8.5, 9.25, 9.25, 0],
				[0, 0, 0, 0, 0]
			]
		]
	])

@pytest.fixture
def backwards_expected_filters_gradient():	# TODO
	return np.array([
		[
			[
				[2208, 2252],
				[2428, 2472]
			],
			[
				[3308, 3352],
				[3528, 3572]
			]
		],
		[
			[
				[3200, 3276],
				[3580, 3656]
			],
			[
				[5100, 5176],
				[5480, 5556]
			]
		]
	])

@pytest.fixture
def backwards_expected_bias_gradient():	# TODO
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
