"""
TC1:
- 5 nodes
- input shape (5,1)
- bias = 0.2 (5,1)
"""
from _pytest.assertion import pytest_sessionfinish
import pytest
from cnn.layers import FC
import numpy as np

from cnn.params import CNNParam

@pytest.fixture
def n():
	return 5

@pytest.fixture
def input_shape():
	return (5,1)

@pytest.fixture
def batch_size():
	return 2

@pytest.fixture
def fc_layer(n,input_shape):
	layer = FC(
		n,
		input_shape=input_shape,
		track_history=False
	)
	layer.prepare_layer()
	layer.weights = np.arange(n*input_shape[0]).reshape((n,input_shape[0]))
	layer.weights.trainable = False
	layer.bias = np.array([[0.2]]*n)
	layer.bias.trainable = False
	return layer

@pytest.fixture
def forwards_input(input_shape,batch_size):
	return np.arange(input_shape[0]*batch_size).reshape((input_shape[0],batch_size))

@pytest.fixture
def forwards_expected_result():
	return np.array(
		[
			[60.2, 70.2],
			[160.2, 195.2],
			[260.2, 320.2],
			[360.2, 445.2],
			[460.2, 570.2]
		]
	)

@pytest.fixture
def backwards_input(forwards_expected_result):
	return np.arange(np.prod(forwards_expected_result.shape)).reshape(forwards_expected_result.shape)

@pytest.fixture
def backwards_expected_output():
	return np.array(
		[
			[ 300, 350 ],
			[ 320, 375 ],
			[ 340, 400 ],
			[ 360, 425 ],
			[ 380, 450 ]
		]
	)

@pytest.fixture
def backwards_expected_weights_gradient():
	return np.array(
		[
			[ 1, 3, 5, 7, 9],
			[ 3, 13, 23, 33, 43],
			[ 5, 23, 41, 59, 77],
			[ 7, 33, 59, 85, 111],
			[ 9, 43, 77, 111, 145]
		]
	)

@pytest.fixture
def backwards_expected_bias_gradient(backwards_input):
	return backwards_input.sum(axis=1,keepdims=True)

def test_param_class_persistance(fc_layer):
	assert isinstance(fc_layer.weights,CNNParam)
	assert isinstance(fc_layer.bias,CNNParam)
	fc_layer.weights = [1,2,3,4]
	fc_layer.bias = [1,2,3]
	assert isinstance(fc_layer.weights,CNNParam)
	assert isinstance(fc_layer.bias,CNNParam)

def test_forwards(fc_layer,forwards_input,forwards_expected_result):

	assert np.array_equal(
		fc_layer._forwards(forwards_input),
		forwards_expected_result
	)

def test_backwards(fc_layer,backwards_input,backwards_expected_output,backwards_expected_weights_gradient,backwards_expected_bias_gradient,forwards_input,forwards_expected_result):
	fc_layer.input = forwards_input
	fc_layer.output = forwards_expected_result

	assert np.array_equal(
		fc_layer._backwards(backwards_input),
		backwards_expected_output
	)
	assert np.array_equal(
		fc_layer.weights.gradient,
		backwards_expected_weights_gradient
	)
	assert np.array_equal(
		fc_layer.bias.gradient,
		backwards_expected_bias_gradient
	)
