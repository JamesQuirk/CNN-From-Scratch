from src.cnn import CNN

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
	assert len(model.structure) == 4	# Flatten Layer added automatically.
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
	assert len(model.structure) == 4
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
	assert L4.next_layer == None

def test_activation():
	pass

if __name__ == '__main__':
	pass