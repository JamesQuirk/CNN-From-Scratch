
from random import sample
import numpy as np
from numpy.core.fromnumeric import size
from src.cnn import CNN
import mnist_dataloader
np.set_printoptions(linewidth=200)

train_images, train_labels, test_images, test_labels = mnist_dataloader.get_data()
train_images = train_images.reshape((train_images.shape[0],1,train_images.shape[1],train_images.shape[2]))
test_images = test_images.reshape((test_images.shape[0],1,test_images.shape[1],test_images.shape[2]))
print(train_images.shape,train_labels.shape)


sampleX = train_images[:5,:,:20,:20]	# first 5 images


def test_vect_conv(X,learning_rate=0.01,filt_shape=(3,3),num_filts=5,stride=2):
	# Params
	input_shape = X.shape[1:]

	model = CNN()
	model.LEARNING_RATE = learning_rate
	conv_layer = CNN.Conv_Layer(filt_shape=filt_shape,num_filters=num_filts,stride=stride,input_shape=input_shape,vectorised=False,track_history=False)
	vect_conv_layer = CNN.Conv_Layer(filt_shape=filt_shape,num_filters=num_filts,stride=stride,input_shape=input_shape,vectorised=True,track_history=False)
	conv_layer.model = model
	vect_conv_layer.model = model
	conv_layer.prepare_layer()
	vect_conv_layer.prepare_layer()
	# ------------- FORWARDS -----------------------
	conv_result = conv_layer._forwards(sampleX)
	vect_conv_result = vect_conv_layer._forwards(sampleX)
	assert np.allclose(conv_result,vect_conv_result), 'Results not equal'

	print('Forwards Test Passed!')

	# ------------- BACKWARDS -----------------------
	ex_cost_gradient = np.random.normal(size=conv_result.shape)
	conv_result_back = conv_layer._backwards(ex_cost_gradient)
	vect_conv_result_back = vect_conv_layer._backwards(ex_cost_gradient)
	assert np.allclose(conv_result_back,vect_conv_result_back), 'Results not equal'
	print('Backwards Test Passed!')

	return True



# -------------------------------- TEST 1 --------------------------------
# test_vect_conv(sampleX,0.01,(5,5),5,2)
# test_vect_conv(sampleX,0.1,(3,5),7,1)

# -------------------------------- TEST 2 --------------------------------

lr_choices = [0.1,0.0001]
filt_dim_choices = [1,5,8]
num_filts_choices = [1,5]
stride_choices = [1,3]

total_tests = len(lr_choices) * (len(filt_dim_choices)**2) * len(num_filts_choices) * len(stride_choices)
print(f'Running {total_tests} tests...')
passed = 0
ti = 0
for lr in lr_choices:
	for filt_rows in filt_dim_choices:
		for filt_cols in filt_dim_choices:
			for num_filts in num_filts_choices:
				for s in stride_choices:
					try:
						test_vect_conv(sampleX.copy(),lr,(filt_rows,filt_cols),num_filts,s)
						passed += 1
						print('Passed test:',ti)
					except AssertionError as err:
						print(f'Failed test >> lr: {lr} | filt_dims: {(filt_rows,filt_cols)} | num filts: {num_filts} | stride: {s}')
						print(err)
					ti += 1
						

print(f'Passed {passed}/{total_tests}')