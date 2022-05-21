"""
TC1:
- filt shape (3,3)
- stride 2
- pool type mean
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
		pool_type='mean',
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
					[6, 8],
					[16, 18]
				],
				[
					[31, 33],
					[41, 43]
				]
			],
			[
				[
					[56, 58],
					[66, 68]
				],
				[
					[81, 83],
					[91, 93]
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
					[0,0,1,1,1],
					[0,0,1,1,1],
					[2,2,6,4,4],
					[2,2,5,3,3],
					[2,2,5,3,3]
				],
				[
					[4,4,9,5,5],
					[4,4,9,5,5],
					[10,10,22,12,12],
					[6,6,13,7,7],
					[6,6,13,7,7]
				]
			],
			[
				[
					[8,8,17,9,9],
					[8,8,17,9,9],
					[18,18,38,20,20],
					[10,10,21,11,11],
					[10,10,21,11,11]
				],
				[
					[12,12,25,13,13],
					[12,12,25,13,13],
					[26,26,54,28,28],
					[14,14,29,15,15],
					[14,14,29,15,15]
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
	print(pool_layer._backwards(backwards_input))
	assert np.array_equal(
		pool_layer._backwards(backwards_input),
		backwards_expected_result
	)

def test_vectorised_backwards(pool_layer,backwards_input,forwards_input,forwards_expected_result,backwards_expected_result):
	pool_layer.VECTORISED = True
	pool_layer._forwards(forwards_input)
	pool_layer.output = forwards_expected_result
	print(backwards_expected_result)
	print(pool_layer._backwards(backwards_input))
	assert np.array_equal(
		pool_layer._backwards(backwards_input),
		backwards_expected_result
	)
