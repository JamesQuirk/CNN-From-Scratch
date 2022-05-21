"""
TC1:
- input (2,5)
"""
import numpy as np
import pytest
from cnn.layers.activations import ReLU

@pytest.fixture
def input_shape():
	return (2,5)

@pytest.fixture
def relu_layer(input_shape):
	layer = ReLU(input_shape=input_shape)
	class DummyPrevLayer:
		output = np.zeros(input_shape)
		OUTPUT_SHAPE = input_shape
	layer.prev_layer = DummyPrevLayer()
	layer.prepare_layer()
	return layer

@pytest.fixture
def forwards_input(input_shape):
	arr = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float)
	median = np.median(arr)
	arr -= median
	return arr

@pytest.fixture
def forwards_expected_result():
	return np.array(
		[
			[0, 0, 0, 0, 0],
			[0.5, 1.5, 2.5, 3.5, 4.5]
		]
	)

@pytest.fixture
def backwards_input(relu_layer):
	out_shape = relu_layer.OUTPUT_SHAPE
	return np.arange(np.prod(out_shape)).reshape(out_shape)

@pytest.fixture
def backwards_expected_result():
	return np.array(
		[
			[0, 0, 0, 0, 0],
			[5, 6, 7, 8, 9]
		]
	)

def test_forwards(relu_layer,forwards_input,forwards_expected_result):
	assert np.array_equal(
		relu_layer._forwards(forwards_input),
		forwards_expected_result
	)

def test_backwards(relu_layer,backwards_input,forwards_input,forwards_expected_result,backwards_expected_result):
	relu_layer.input = forwards_input
	relu_layer.output = forwards_expected_result
	assert np.array_equal(
		relu_layer._backwards(backwards_input),
		backwards_expected_result
	)
