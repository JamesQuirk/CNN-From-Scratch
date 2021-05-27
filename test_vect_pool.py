
import numpy as np
from src.cnn import CNN
import mnist_dataloader
np.set_printoptions(linewidth=200)

train_images, train_labels, test_images, test_labels = mnist_dataloader.get_data()
train_images = train_images.reshape((train_images.shape[0],1,train_images.shape[1],train_images.shape[2]))
test_images = test_images.reshape((test_images.shape[0],1,test_images.shape[1],test_images.shape[2]))
print(train_images.shape,train_labels.shape)


sampleX = train_images[:5,:,:10,:10]	# first 5 images
filter_shape = (2,2)

pool_layer = CNN.Pool_Layer(filter_shape,stride=2,pool_type='max',input_shape=(1,10,10),vectorised=False,track_history=False)
vect_pool_layer = CNN.Pool_Layer(filter_shape,stride=2,pool_type='max',input_shape=(1,10,10),vectorised=True,track_history=False)
pool_layer.prepare_layer()
vect_pool_layer.prepare_layer()
# ------------- FORWARDS -----------------------
pool_result = pool_layer._forwards(sampleX)
vect_pool_result = vect_pool_layer._forwards(sampleX)
assert np.array_equal(pool_result,vect_pool_result), 'Results not equal'

print('Test Passed!')

# ------------- BACKWARDS -----------------------
assert np.array_equal(pool_layer._backwards(pool_result),vect_pool_layer._backwards(pool_result)), 'Results not equal'
print('Test Passed!')
