from src.cnn import CNN
import mnist
import numpy as np
np.set_printoptions(linewidth=200)


def get_data():
	print('Loading data...')
	mndata = mnist.MNIST('./data',return_type='numpy')
	mndata.gz = False

	train_images, train_labels = mndata.load_training()

	test_images, test_labels = mndata.load_testing()

	# reshape to square
	train_images = train_images.reshape((len(train_images),28,28)) / 255
	train_labels = train_labels.reshape((len(train_labels),1))

	test_images = test_images.reshape((len(test_images),28,28)) / 255
	test_labels = test_labels.reshape((1,len(test_labels)))

	# labels need to be 'one-hot encoded'
	train_labels = CNN.one_hot_encode(train_labels,10)
	test_labels = CNN.one_hot_encode(test_labels,10)

	print('Train images shape:', train_images.shape, 'Train labels shape:', train_labels.shape)
	print('Test images shape:', test_images.shape, 'Test labels shape:', test_labels.shape)

	return train_images, train_labels, test_images, test_labels

train_images, train_labels, test_images, test_labels = get_data()

model = CNN((1,28,28),optimiser_method='adam')

# model.add_layer(
# 	CNN.Conv_Layer(filt_shape=(5,5),num_filters=5,stride=2,pad_type='include')
# )
# model.add_layer(
# 	CNN.Pool_Layer(filt_shape=(3,3),stride=1,pool_type='max',pad_type='include')
# )
model.add_layer(
	CNN.Flatten_Layer()
)
model.add_layer(
	CNN.FC_Layer(500,activation='relu',initiation_method='kaiming_normal')
)
model.add_layer(
	CNN.FC_Layer(200,activation='relu',initiation_method='kaiming_normal')
)
model.add_layer(
	CNN.FC_Layer(10,activation='softmax',initiation_method='xavier_normal')
)
model.prepare_model()

input('Run training?')

train_finish, training_duration = model.train(
	train_images,
	train_labels,
	epochs=20,
	max_batch_size=6000,
	shuffle=False,
	cost_fn='cross_entropy',
	learning_rate=0.001
)
print(f'Training duration: {training_duration}')
save_name = f'nn_model_adam_lr0-001_{train_finish.strftime("%H-%M-%S")}.pkl'
print(f'Model saved to: {save_name}')
model.save_model(save_name)
