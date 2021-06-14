"""
Configuration NN-128-128-10 scored 88.2% train accuracy and 90.3% test accuracy


"""


from cnn import optimisers
import cnn
import mnist_dataloader
# np.set_printoptions(linewidth=200)


train_images, train_labels, test_images, test_labels = mnist_dataloader.get_data()
train_images = train_images.reshape((train_images.shape[0],1,train_images.shape[1],train_images.shape[2]))
test_images = test_images.reshape((test_images.shape[0],1,test_images.shape[1],test_images.shape[2]))
print(train_images.shape,train_labels.shape)

model = cnn.Model()

# model.add_layer(
# 	CNN.Conv_Layer(filt_shape=(5,5),num_filters=5,stride=2,pad_type='include',input_shape=(1,28,28))
# )
# model.add_layer(
# 	CNN.Pool_Layer(filt_shape=(3,3),stride=1,pool_type='max',pad_type='include')
# )
model.add_layer(
	cnn.layers.Flatten(input_shape=(1,28,28))
)
model.add_layer(
	cnn.layers.FC(128,activation='relu',initiation_method='kaiming_normal')
)
model.add_layer(
	cnn.layers.FC(128,activation='relu',initiation_method='kaiming_normal')
)
model.add_layer(
	cnn.layers.FC(10,activation='softmax',initiation_method='kaiming_normal')
)
model.prepare_model(optimiser=optimisers.Adam())

input('Run training?')

train_finish, training_duration = model.train(
	train_images,
	train_labels,
	epochs=3,
	max_batch_size=5000,
	shuffle=False,
	cost_fn='cross_entropy',
	learning_rate=0.001,
	random_seed=42
)
print(f'Test Accuracy: {model.evaluate(test_images,test_labels)*100:.2f}%')

print(f'Training duration: {training_duration}')
save_name = f'models/nn_model_adam_tf_comparitor_vectorised_{train_finish.strftime("%H-%M-%S")}.pkl'
print(f'Model saved to: {save_name}')
model.save_model(save_name)

