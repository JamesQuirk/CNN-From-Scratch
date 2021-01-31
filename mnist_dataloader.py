import mnist
import numpy as np
from src.cnn import CNN
np.set_printoptions(linewidth=200)


def get_data(normalise=True,one_hot=True):
	print('Loading data...')
	mndata = mnist.MNIST('./data',return_type='numpy')
	mndata.gz = False

	train_images, train_labels = mndata.load_training()

	test_images, test_labels = mndata.load_testing()

	normalising_factor = 255 if normalise else 1

	# reshape to square
	train_images = train_images.reshape((len(train_images),28,28)) / normalising_factor
	train_labels = train_labels.reshape((len(train_labels),1))

	test_images = test_images.reshape((len(test_images),28,28)) / normalising_factor
	test_labels = test_labels.reshape((1,len(test_labels)))

	# labels need to be 'one-hot encoded'
	train_labels = CNN.one_hot_encode(train_labels,10) if one_hot else train_labels
	test_labels = CNN.one_hot_encode(test_labels,10) if one_hot else test_labels

	print('Train images shape:', train_images.shape, 'Train labels shape:', train_labels.shape)
	print('Test images shape:', test_images.shape, 'Test labels shape:', test_labels.shape)

	return train_images, train_labels, test_images, test_labels
