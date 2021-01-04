from src.cnn import CNN
import numpy as np

def test_add_layer_1():
	model = CNN((3,12,12))
	model.add_layer(
		CNN.Conv_Layer(filt_shape=(3,3),num_filters=2)
	)
	assert len(model.structure) == 1
	model.add_layer(
		CNN.Pool_Layer(filt_shape=(3,3),stride=3)
	)
	assert len(model.structure) == 2
	model.add_layer(
		CNN.FC_Layer(9)
	)
	assert len(model.structure) == 5	# Flatten and Activation Layer added automatically.
	print(model.structure)

def test_add_layer_2():
	model = CNN((3,12,12))
	model.add_layer(
		CNN.Conv_Layer(filt_shape=(3,3),num_filters=2)
	)
	assert len(model.structure) == 1
	model.add_layer(
		CNN.Pool_Layer(filt_shape=(3,3),stride=3)
	)
	assert len(model.structure) == 2
	model.add_layer(
		CNN.Flatten_Layer()
	)
	assert len(model.structure) == 3	# Flatten Layer added manually.
	model.add_layer(
		CNN.FC_Layer(9)
	)
	assert len(model.structure) == 5	# Activation Layer added automatically.
	print(model.structure)

def test_prepare_model():
	model = CNN((3,12,12))
	L1 = CNN.Conv_Layer(filt_shape=(3,3),num_filters=2)
	L2 = CNN.Pool_Layer(filt_shape=(3,3),stride=3)
	L3 = CNN.Flatten_Layer()
	L4 = CNN.FC_Layer(9)
	model.add_layer(
		L1
	)
	model.add_layer(
		L2
	)
	model.add_layer(
		L3
	)
	model.add_layer(
		L4
	)

	model.prepare_model()

	assert L1.output.shape == (2,10,10)
	assert L1.filters.shape == (2,3,3,3)

	assert L2.output.shape == (2,3,3)

	assert L3.output.shape == (18,1)

	assert L4.output.shape == (9,1)

	assert L1.prev_layer == None
	assert L1.next_layer == L2
	assert L2.prev_layer == L1
	assert L2.next_layer == L3
	assert L3.prev_layer == L2
	assert L3.next_layer == L4
	assert L4.prev_layer == L3
	assert L4.next_layer.LAYER_TYPE == 'ACTIVATION'

def test_activation_relu():
	arr1 = np.arange(-9,9)
	arr2 = arr1.reshape((2,3,3))

	assert ( CNN.activation(arr1, activation='relu',derivative=False) == np.array([0,0,0,0,0,0,0,0,0,0,1,2,3,4,5,6,7,8]) ).all()

	assert (CNN.activation(arr2, activation='relu',derivative=False) == np.array([
			[
				[0,0,0],
				[0,0,0],
				[0,0,0]
			],
			[
				[0,1,2],
				[3,4,5],
				[6,7,8]
			]
		]
	)).all()

	assert ( CNN.activation(arr1, activation='relu',derivative=True) == np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]) ).all()

	assert (CNN.activation(arr2, activation='relu', derivative=True) == np.array(
		[
			[
				[0,0,0],
				[0,0,0],
				[0,0,0]
			],
			[
				[0,1,1],
				[1,1,1],
				[1,1,1]
			]
		]
	)).all()

def test_activation_softmax():
	# TODO: arrays not correct yet - calculate expected output.
	arr1 = np.arange(-9,9)
	arr2 = arr1.reshape((2,3,3))

	assert ( CNN.activation(arr1, activation='softmax',derivative=False) == np.array([2.61693978e-08,7.11357986e-08,1.93367149e-07, 5.25626407e-07,1.42880071e-06,3.88388301e-06,1.05574886e-05,2.86982294e-05,7.80098755e-05, 2.12052827e-04,5.76419346e-04,1.56687023e-03,4.25919489e-03,1.15776921e-02, 3.14714300e-02, 8.55482162e-02,2.32544161e-01,6.32120568e-01]) ).all()

	assert ( CNN.activation(arr2,activation='softmax',derivative=False) == np.array([
		[
			[2.61693978e-08,7.11357986e-08,1.93367149e-07],
			[5.25626407e-07,1.42880071e-06,3.88388301e-06],
			[1.05574886e-05,2.86982294e-05,7.80098755e-05]
		],
       [
			[2.12052827e-04,5.76419346e-04,1.56687023e-03],
			[4.25919489e-03,1.15776921e-02,3.14714300e-02],
			[8.55482162e-02,2.32544161e-01,6.32120568e-01]
		]
	]) ).all()

	assert ( CNN.activation(arr1, activation='softmax',derivative=True) == np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]) ).all()

	assert (CNN.activation(arr2, activation='softmax', derivative=True) == np.array(
		[
			[
				[0,0,0],
				[0,0,0],
				[0,0,0]
			],
			[
				[0,1,1],
				[1,1,1],
				[1,1,1]
			]
		]
	)).all()

if __name__ == '__main__':
	pass