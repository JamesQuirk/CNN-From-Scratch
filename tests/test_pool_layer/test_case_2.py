"""
TC1:
- filt shape (3,3)
- stride 2
- pool type min
- input shape (2,5,5)
"""
import pytest
from cnn.layers import Pool
import numpy as np

@pytest.fixture
def input_shape():
	return (2,5,5)

@pytest.fixture
def pool_layer():
	layer = Pool(
		filt_shape=(3,3),
		stride=2,
		pool_type='min',
		input_shape=(2,5,5),
		vectorised=False,
		track_history=False
	)
	layer.prepare_layer()
	return layer

@pytest.fixture
def forwards_input(input_shape):
	batch_size = 2
	return np.arange(batch_size*np.prod(input_shape)).reshape((batch_size,*input_shape))

@pytest.fixture
def forwards_expected_result():
	return np.array(
		[
			[
				[
					[0, 2],
					[10, 12]
				],
				[
					[25, 27],
					[35, 37]
				]
			],
			[
				[
					[50, 52],
					[60, 62]
				],
				[
					[75, 77],
					[85, 87]
				]
			]
		]
	)

@pytest.fixture
def backwards_input(forwards_expected_result):
	return np.arange(np.prod(forwards_expected_result.shape)).reshape((forwards_expected_result.shape))

@pytest.fixture
def backwards_expected_result():
	return np.array(
		[
			[
				[
					[0,0,1,0,0],
					[0,0,0,0,0],
					[2,0,3,0,0],
					[0,0,0,0,0],
					[0,0,0,0,0]
				],
				[
					[4,0,5,0,0],
					[0,0,0,0,0],
					[6,0,7,0,0],
					[0,0,0,0,0],
					[0,0,0,0,0]
				]
			],
			[
				[
					[8,0,9,0,0],
					[0,0,0,0,0],
					[10,0,11,0,0],
					[0,0,0,0,0],
					[0,0,0,0,0]
				],
				[
					[12,0,13,0,0],
					[0,0,0,0,0],
					[14,0,15,0,0],
					[0,0,0,0,0],
					[0,0,0,0,0]
				]
			]
		]
	)

def test_forwards(pool_layer,forwards_input,forwards_expected_result):
	assert np.array_equal(
		pool_layer._forwards(forwards_input),
		forwards_expected_result
	)

def test_vectorised_forwards(pool_layer,forwards_input,forwards_expected_result):
	pool_layer.VECTORISED = True
	assert np.array_equal(
		pool_layer._forwards(forwards_input),
		forwards_expected_result
	)

def test_backwards(pool_layer,backwards_input,forwards_input,forwards_expected_result,backwards_expected_result):
	pool_layer.input = forwards_input
	pool_layer.padded_input = forwards_input
	pool_layer.output = forwards_expected_result
	assert np.array_equal(
		pool_layer._backwards(backwards_input),
		backwards_expected_result
	)

def test_vectorised_backwards(pool_layer,backwards_input,forwards_input,forwards_expected_result,backwards_expected_result):
	pool_layer.VECTORISED = True
	pool_layer._forwards(forwards_input)
	pool_layer.output = forwards_expected_result
	assert np.array_equal(
		pool_layer._backwards(backwards_input),
		backwards_expected_result
	)
